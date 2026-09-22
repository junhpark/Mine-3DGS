"""An image-only dataset, built from a registered SfM reconstruction (§Phase 3 C3).

The output is the *same* dataset contract every other path produces — ``manifest.json``,
``sparse/0``, ``images/``, ``init_points.ply``, optional ``masks/`` and ``centerline.csv``. No
fork, no second schema, no "video mode" in the trainer. What differs is where the geometry came
from, and that difference is recorded rather than declared:

* ``source`` is ``video`` or ``video360``, never ``tls``. The TLS value would bypass both of
  the gates that exist for image-derived data (``Manifest.validate`` and the protocol judge),
  which is the single cheapest way this path could be made to lie.
* ``initialization.source`` is ``sfm_sparse``, and ``init_points.ply`` is built from the
  reconstruction's own points. That is checked, not believed: the manifest records the SfM
  model's digest and the id it came from, and :func:`check_image_only_dataset` rebuilds the
  binding. Copying a TLS cloud in and writing ``sfm_sparse`` above it does not survive.
* ``scale.basis`` is the registration's basis, and the registration block carries the record's
  id, digest and support extent so the protocol judge can decide overlap with this dataset's
  own holdout.

The measured Sim(3) is applied exactly once, here: ``SFM_INTERNAL → TLS_GLOBAL → LOCAL_METRIC``
for cameras, points and the initialisation cloud together. Nothing downstream re-applies it,
which is what keeps the cameras and the geometry in the same world.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

import minegs
from minegs.core.centerline import Centerline
from minegs.core.config import VersionedModel, config_hash
from minegs.core.errors import ContractError
from minegs.core.frames import SE3, Frame, Sim3, rotmat_to_quat
from minegs.core.manifest import (
    CaptureGroup,
    CenterlineRef,
    CoordinateFrames,
    GeometryHoldout,
    Initialization,
    Manifest,
    ManifestProvenance,
    PanoConvention,
    Registration,
    Scale,
    Split,
    spans_overlap,
)
from minegs.core.pointcloud import PointCloud, voxel_downsample, write_ply
from minegs.core.provenance import git_commit, sha256_file, tool_versions
from minegs.eval.register.models import (
    CLAIM_BEARING_FIELDS,
    RegistrationRecord,
    check_registration,
    claim_bearing_values,
    load_registration,
)
from minegs.ingest.common import colmap_io
from minegs.ingest.video.models import check_frameset, load_frameset
from minegs.ingest.video.sfm.models import SPARSE_FILES, check_sfm, load_sfm, model_digest

CENTERLINE_FILE = "centerline.csv"
#: Immutable copies of the three records this dataset rests on. They live *inside* the dataset
#: so that they are inside its hash: a manifest that points at evidence living outside the
#: hashed tree can be paired with different evidence afterwards and nothing notices.
PROVENANCE_DIR = "provenance/phase3"
#: The selected SFM_INTERNAL model, kept inside the dataset tree (and so inside its hash).
SFM_MODEL_DIR = "sfm_model"


class SfmDatasetConfig(VersionedModel):
    """What the builder needs that the records do not already say."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    dataset_id: str = Field(min_length=1)
    #: Chosen by the operator, and checked against the frame set's kind.
    source: Literal["video", "video360"] = "video"
    #: Images per capture group on the plain-video path. A 360 set groups by panorama instead,
    #: because the crops of one frame share an optical centre and are one observation.
    group_size: int = Field(default=8, ge=1)
    test_groups: list[str] = Field(default_factory=list)
    geometry_holdout_m: list[tuple[float, float]] = Field(default_factory=list)
    #: False is the reconstruction test: the holdout's images are trained on and the held-out
    #: *TLS geometry* is what the result is measured against. True is the extrapolation test,
    #: and it is a much stronger statement about the experiment, so it is not the default — a
    #: dataset should not end up labelled an extrapolation test because nobody passed a flag.
    #: When it is true the builder takes the holdout's capture groups out of ``train_groups``,
    #: so the manifest states what happened rather than what was asked for.
    holdout_images_excluded: bool = False
    centerline_file: str | None = None
    centerline_source: Literal["design", "extracted"] = "extracted"
    centerline_bin_m: float = Field(default=2.0, gt=0)
    init_voxel_m: float | None = Field(default=0.02, gt=0)
    init_max_points: int = Field(default=1_000_000, ge=1)
    sparse_max_points: int = Field(default=200_000, ge=0)
    seed: int = 0
    local_origin_rounding_m: float = Field(default=0.1, gt=0)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def transform_model(model: colmap_io.ColmapModel, T: Sim3) -> colmap_io.ColmapModel:
    """Move a COLMAP model into another frame through a similarity.

    A camera pose is rigid, so the scale cannot go into the rotation: the camera *centre*
    moves with the similarity and the orientation turns with its rotation. Writing the scale
    into the extrinsics instead is the classic way to produce a model that reprojects
    beautifully and is the wrong size.
    """
    from minegs.core.frames import quat_to_rotmat

    R_s, s, t_s = T.R, T.s, T.t
    images = {}
    for iid, im in model.images.items():
        R_cw = quat_to_rotmat(im.qvec)
        centre = -R_cw.T @ im.tvec
        centre_new = s * (R_s @ centre) + t_s
        R_new = R_cw @ R_s.T
        images[iid] = colmap_io.Image(
            id=im.id,
            qvec=rotmat_to_quat(R_new),
            tvec=-R_new @ centre_new,
            camera_id=im.camera_id,
            name=im.name,
            xys=im.xys,
            point3D_ids=im.point3D_ids,
        )
    points = {
        pid: colmap_io.Point3D(p.id, T.apply(p.xyz), p.rgb, p.error, p.image_ids, p.point2D_idxs)
        for pid, p in model.points3D.items()
    }
    return colmap_io.ColmapModel(
        dict(model.cameras), images, points, dict(model.rigs), dict(model.frames)
    )


def _camera_centres(model: colmap_io.ColmapModel) -> dict[str, np.ndarray]:
    from minegs.core.frames import quat_to_rotmat

    out = {}
    for im in model.images.values():
        R = quat_to_rotmat(im.qvec)
        out[im.name] = -R.T @ im.tvec
    return out


def _groups(
    cfg: SfmDatasetConfig, names: list[str], crops_by_parent: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Capture groups: one per panorama for 360, otherwise consecutive chunks of the traverse.

    Groups are what the split and the leak checks operate on, so they have to mean something
    physical. The crops of one panorama share an optical centre and are one observation of one
    place; along a corridor, a run of consecutive frames is the nearest equivalent.
    """
    if crops_by_parent:
        return {f"pano_{Path(parent).stem}": sorted(v) for parent, v in crops_by_parent.items()}
    out: dict[str, list[str]] = {}
    for i in range(0, len(names), cfg.group_size):
        out[f"seg_{i // cfg.group_size:04d}"] = names[i : i + cfg.group_size]
    return out


def build_dataset_from_sfm(
    frameset_dir: str | Path,
    sfm_dir: str | Path,
    registration_dir: str | Path,
    out_dir: str | Path,
    cfg: SfmDatasetConfig,
    *,
    overwrite: bool = False,
) -> tuple[Manifest, Path]:
    """Turn a registered reconstruction into a dataset the existing training path accepts."""
    rec_fs, fs_dir = load_frameset(frameset_dir)
    check_frameset(rec_fs, fs_dir)
    rec_sfm, sfm_root = load_sfm(sfm_dir)
    model_dir = check_sfm(rec_sfm, sfm_root, images_sha256=rec_fs.images_sha256)
    rec_reg, reg_root = load_registration(registration_dir)
    if rec_reg.sfm_id != rec_sfm.sfm_id:
        raise ContractError(
            f"registration {rec_reg.registration_id} is of SfM {rec_reg.sfm_id}, not "
            f"{rec_sfm.sfm_id}: a measured transform belongs to one reconstruction"
        )
    from minegs.eval.register.models import check_registration

    check_registration(rec_reg, rec_sfm.model_sha256)
    if rec_sfm.frameset_id != rec_fs.frameset_id:
        raise ContractError(
            f"SfM {rec_sfm.sfm_id} was reconstructed from frame set {rec_sfm.frameset_id}, not "
            f"{rec_fs.frameset_id}"
        )
    expected_source = "video360" if rec_fs.kind == "video360" else "video"
    if cfg.source != expected_source:
        raise ContractError(
            f"the frame set is a {rec_fs.kind} set, so this dataset's source is "
            f"{expected_source!r}, not {cfg.source!r}"
        )

    ds = Path(out_dir)
    if ds.exists() and any(ds.iterdir()):
        if not overwrite:
            raise ContractError(
                f"{ds} is not empty; a dataset is built into a directory of its own"
            )
        shutil.rmtree(ds)
    ds.mkdir(parents=True, exist_ok=True)

    # ---- one application of the measured transform, for everything at once
    model_sfm = colmap_io.read_model(model_dir)
    T_tls_from_sfm = rec_reg.sim3()
    model_tls = transform_model(model_sfm, T_tls_from_sfm)
    points_tls = model_tls.points_xyz()
    centres_tls = np.array(list(_camera_centres(model_tls).values()))
    origin = np.round(centres_tls.mean(axis=0) / cfg.local_origin_rounding_m) * (
        cfg.local_origin_rounding_m
    )
    T_tls_from_local = SE3(np.eye(3), origin)
    model_local = transform_model(model_tls, Sim3.from_se3(T_tls_from_local.inverse()))

    # ---- images and masks, copied from the frame set (these bytes, not a re-glob)
    (ds / "images").mkdir(parents=True)
    for entry in rec_fs.images:
        target = ds / "images" / entry.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rec_fs.image_root(fs_dir) / entry.name, target)
    if rec_fs.masks:
        mask_root = rec_fs.mask_root(fs_dir)
        assert mask_root is not None
        for mask in rec_fs.masks:
            target = ds / "masks" / mask.mask_file
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_root / mask.mask_file, target)

    # ---- the reference axis, derived before anything is excluded along it
    # From the *whole* reconstruction: a centerline extracted from the kept part only would
    # have a hole where the holdout is, and every chainage past it would be renumbered.
    centerline_tls: Centerline | None = None
    if cfg.centerline_file is not None:
        centerline_tls = Centerline.from_csv(
            cfg.centerline_file, "TLS_GLOBAL", cfg.centerline_source
        )
    elif len(points_tls) >= 2:
        centerline_tls = Centerline.extract_from_points(
            points_tls, bin_m=cfg.centerline_bin_m, frame="TLS_GLOBAL"
        )
    if centerline_tls is not None:
        centerline_tls.to_csv(ds / CENTERLINE_FILE)

    # ---- the holdout, taken out of the geometry rather than only declared
    #
    # `initialization.excluded_chainage_ranges_m` is a statement that the held-out chainage is
    # not in what the model starts from, and on this path the initialisation *is* the
    # reconstruction: leaving its points in would hand the model the structure triangulated
    # from the very frames the holdout keeps out of training, under a manifest saying it had
    # not. Both the init cloud and `sparse/0` are filtered, because the trainer can be told to
    # initialise from either (`stage_dataset(use_init_points=False)`), and an exclusion that
    # one of two paths honours is not an exclusion.
    holdout = [(float(lo), float(hi)) for lo, hi in cfg.geometry_holdout_m]
    excluded_ids: set[int] = set()
    if holdout:
        if centerline_tls is None:
            raise ContractError(
                "a geometry holdout is declared in chainage, but this dataset has no reference "
                "axis: without one there is no way to say which reconstruction points are "
                "inside the holdout, so the exclusion could only be asserted"
            )
        ids = sorted(model_tls.points3D)
        s = np.atleast_1d(
            centerline_tls.project(np.array([model_tls.points3D[i].xyz for i in ids]))[0]
        )
        inside = np.zeros(len(ids), bool)
        for lo, hi in holdout:
            inside |= (s >= lo) & (s <= hi)
        excluded_ids = {ids[k] for k in np.flatnonzero(inside)}
        model_local.points3D = {
            k: v for k, v in model_local.points3D.items() if k not in excluded_ids
        }
        if not model_local.points3D:
            raise ContractError(
                f"every reconstruction point falls inside the declared holdout {holdout}; "
                "there would be nothing left to initialise training from"
            )

    # ---- initialisation from the reconstruction's own points, never from a reference cloud
    init_xyz = model_local.points_xyz()
    if cfg.init_voxel_m:
        init_xyz = init_xyz[voxel_downsample(init_xyz, cfg.init_voxel_m)]
    rng = np.random.default_rng(cfg.seed)
    if len(init_xyz) > cfg.init_max_points:
        idx = np.sort(rng.choice(len(init_xyz), cfg.init_max_points, replace=False))
        init_xyz = init_xyz[idx]
    if len(init_xyz) == 0:
        raise ContractError(
            "the reconstruction has no points left after downsampling; there is nothing to "
            "initialise training from"
        )
    init = PointCloud(
        init_xyz,
        rgb=np.full((len(init_xyz), 3), 128, np.uint8),
        frame=Frame.LOCAL_METRIC.value,
    )
    write_ply(init, ds / "init_points.ply")

    # ---- sparse model: the reconstruction, moved. Not a second copy of anything else.
    n_sparse = min(cfg.sparse_max_points, len(model_local.points3D))
    if n_sparse < len(model_local.points3D):
        keep = set(
            int(i) for i in rng.choice(sorted(model_local.points3D), size=n_sparse, replace=False)
        )
        model_local.points3D = {k: v for k, v in model_local.points3D.items() if k in keep}
    colmap_io.write_model(model_local, ds / "sparse" / "0")

    # ---- groups, split, chainage
    crops_by_parent: dict[str, list[str]] = {}
    for crop in rec_fs.crops:
        crops_by_parent.setdefault(crop.parent_frame, []).append(crop.name)
    names = [e.name for e in rec_fs.images]
    grouped = _groups(cfg, names, crops_by_parent)
    centres = _camera_centres(model_tls)
    chainage: dict[str, float] = {}
    spans: dict[str, tuple[float, float]] = {}
    if centerline_tls is not None:
        for gid, members in grouped.items():
            pts = np.array([centres[n] for n in members if n in centres])
            if len(pts):
                # The members' own chainages, not the chainage of their mean position. A group
                # of the plain-video path is a *run* of the traverse and covers a span of
                # drift; collapsing it to one number hides exactly the case that matters — a
                # group whose middle sits outside the holdout while one end reaches inside it.
                # For a 360 group the crops share an optical centre, so the span is a point and
                # this costs nothing.
                s = np.atleast_1d(centerline_tls.project(pts)[0])
                spans[gid] = (float(s.min()), float(s.max()))
                chainage[gid] = float(s.mean())
    unknown = sorted(set(cfg.test_groups) - set(grouped))
    if unknown:
        raise ContractError(f"test groups {unknown} are not capture groups of this dataset")
    train_groups = [g for g in sorted(grouped) if g not in cfg.test_groups]
    if holdout and cfg.holdout_images_excluded:
        # `images_excluded=True` says the holdout's images were not trained on. `train_images()`
        # enforces that at read time only for groups whose chainage is known, and keeps a group
        # with no chainage — so the declaration could be true of the reader and false of the
        # data. Both halves are closed here: a group inside the holdout leaves `train_groups`,
        # and a group nobody could place refuses the build rather than being quietly trained on.
        unplaced = sorted(g for g in train_groups if g not in spans)
        if unplaced:
            raise ContractError(
                f"holdout_images_excluded is set, but capture groups {unplaced[:5]} have no "
                "chainage, so there is no way to say whether their images are inside the "
                "holdout. An exclusion nobody can check is not an exclusion."
            )
        # Overlap of the group's span with the holdout, through the same helper the manifest's
        # own `train_images()` uses, so the build-time exclusion and the read-time one cannot
        # disagree about what "inside" means. A group with one frame in the holdout is a group
        # that saw the holdout.
        inside = [g for g in train_groups if spans_overlap(spans[g], holdout)]
        train_groups = [g for g in train_groups if g not in set(inside)]
        if not train_groups:
            raise ContractError(
                f"every training capture group falls inside the holdout {holdout}; excluding "
                "their images leaves nothing to train on"
            )

    groups = {
        gid: CaptureGroup(
            type="camera_rig" if crops_by_parent else "trajectory_segment",
            members=members,
            chainage_m=chainage.get(gid),
            chainage_range_m=spans.get(gid),
        )
        for gid, members in grouped.items()
    }

    # ---- provenance copies, inside the tree so they are inside the hash
    prov = ds / PROVENANCE_DIR
    prov.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fs_dir / "frameset.json", prov / "frameset.json")
    shutil.copy2(sfm_root / "sfm.json", prov / "sfm.json")
    shutil.copy2(reg_root / "registration.json", prov / "registration.json")
    # The model itself, not only the record of it. Without these bytes the claim "the
    # initialisation is this reconstruction's own points" can only be checked against another
    # copy of the same dataset's geometry — which agrees with itself however it was made.
    # A sparse model is small; this is the cheapest provenance in the tree.
    (prov / SFM_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    for name in SPARSE_FILES:
        shutil.copy2(model_dir / name, prov / SFM_MODEL_DIR / name)

    manifest = Manifest(
        dataset_id=cfg.dataset_id,
        coordinate_frames=CoordinateFrames(T_tls_from_local=T_tls_from_local.to_list()),
        capture_groups=groups,
        split=Split(
            train_groups=train_groups,
            test_groups=list(cfg.test_groups),
            geometry_holdout=GeometryHoldout(
                chainage_ranges_m=holdout,
                points_excluded=True,
                images_excluded=cfg.holdout_images_excluded,
            )
            if holdout
            else None,
        ),
        initialization=Initialization(
            source="sfm_sparse",
            file="init_points.ply",
            groups=train_groups,
            # True of the cloud on disk, not only of this field: the points inside these
            # ranges were dropped above, before the init cloud and `sparse/0` were written.
            excluded_chainage_ranges_m=holdout,
            n_points=len(init),
        ),
        provenance=ManifestProvenance(
            minegs_version=minegs.__version__,
            git_commit=git_commit(),
            config_hash=config_hash(cfg.model_dump(mode="json")),
            tool_versions=tool_versions(),
        ),
        source=cfg.source,
        scale=Scale(
            basis="known_target" if rec_reg.basis == "known_target" else "sim3_to_tls",
            factor=float(rec_reg.scale),
        ),
        registration=Registration(
            method=rec_reg.diagnostics.method,
            scale=float(rec_reg.scale),
            rmse_m=float(rec_reg.diagnostics.rmse_m),
            inlier_ratio=float(rec_reg.diagnostics.inlier_ratio),
            transform=rec_reg.T_tls_from_sfm,
            n_correspondences=rec_reg.diagnostics.n_source,
            inlier_threshold_m=rec_reg.diagnostics.inlier_threshold_m,
            registration_id=rec_reg.registration_id,
            record_sha256=sha256_file(reg_root / "registration.json"),
            sfm_id=rec_sfm.sfm_id,
            basis=rec_reg.basis,
            support_ranges_m=rec_reg.support_ranges_m,
            claim_allowed=rec_reg.claim_allowed,
            claim_refusals=list(rec_reg.claim_refusals),
        ),
        centerline=CenterlineRef(
            file=CENTERLINE_FILE, source=cfg.centerline_source, frame="TLS_GLOBAL"
        )
        if centerline_tls is not None
        else None,
    )
    if rec_fs.pano_convention is not None:
        pc = rec_fs.pano_convention
        manifest.pano_convention = PanoConvention(
            az_sign=pc.az_sign,
            el_flip=pc.el_flip,
            az_offset=pc.az_offset_deg,
            source=pc.source,
            vendor=pc.vendor,
        )
    _write_sfm_provenance(
        ds, rec_fs, rec_sfm, rec_reg, cfg, n_holdout_points_excluded=len(excluded_ids)
    )
    manifest.save_dataset(ds)
    check_image_only_dataset(ds)
    return manifest, ds


def _write_sfm_provenance(
    ds: Path,
    rec_fs: Any,
    rec_sfm: Any,
    rec_reg: Any,
    cfg: SfmDatasetConfig,
    *,
    n_holdout_points_excluded: int = 0,
) -> None:
    """The one file that ties the dataset's init cloud to the reconstruction it came from."""
    (ds / PROVENANCE_DIR / "init_provenance.json").write_text(
        json.dumps(
            {
                "frameset_id": rec_fs.frameset_id,
                "images_sha256": rec_fs.images_sha256,
                "sfm_id": rec_sfm.sfm_id,
                "sfm_model_sha256": rec_sfm.model_sha256,
                "real_sfm_execution": rec_sfm.real_sfm_execution,
                "frame_extraction_real": rec_fs.extraction.real_execution,
                "registration_id": rec_reg.registration_id,
                "registration_basis": rec_reg.basis,
                "support_ranges_m": rec_reg.support_ranges_m,
                "claim_allowed": rec_reg.claim_allowed,
                "init_source": "sfm_sparse",
                "init_voxel_m": cfg.init_voxel_m,
                "holdout_ranges_m": [list(r) for r in cfg.geometry_holdout_m],
                "n_holdout_points_excluded": n_holdout_points_excluded,
                "config": cfg.model_dump(mode="json"),
            },
            indent=2,
        )
        + "\n"
    )


def check_image_only_dataset(dataset_dir: str | Path) -> None:
    """Re-derive what an image-only dataset declares about itself.

    The declaration ``initialization.source = sfm_sparse`` is free to write. What is not free
    is producing an initialisation cloud that is actually the reconstruction's points: this
    re-reads the recorded SfM model, moves it with the recorded transform, and checks that the
    cloud on disk is a subset of the result. A TLS cloud with ``sfm_sparse`` written above it
    does not pass.
    """
    ds = Path(dataset_dir)
    manifest = Manifest.load_dataset(ds, strict_layout=False)
    if manifest.source not in ("video", "video360"):
        return
    prov_dir = ds / PROVENANCE_DIR
    prov_file = prov_dir / "init_provenance.json"
    if not prov_file.is_file():
        raise ContractError(
            f"{ds}: an image-derived dataset with no {PROVENANCE_DIR}/init_provenance.json. "
            "Nothing ties its initialisation to a reconstruction, so `sfm_sparse` is a word."
        )
    prov = json.loads(prov_file.read_text())
    if manifest.initialization.source != "sfm_sparse":
        raise ContractError(
            f"{ds}: source is {manifest.source} but initialization.source is "
            f"{manifest.initialization.source!r}. An image-only dataset initialises from its "
            "own reconstruction; anything else is geometry from somewhere the images never saw."
        )
    init_path = (ds / manifest.initialization.file).resolve()
    if not init_path.is_relative_to(ds.resolve()):
        raise ContractError(
            f"{ds}: initialization.file points outside the dataset ({manifest.initialization.file}). "
            "The dataset hash covers this tree; an initialisation cloud outside it can be "
            "swapped afterwards without changing a single recorded digest."
        )
    sfm_json = prov_dir / "sfm.json"
    if not sfm_json.is_file():
        raise ContractError(f"{ds}: no copy of the SfM record under {PROVENANCE_DIR}")
    from minegs.ingest.video.sfm.models import SfmRecord

    rec_sfm = SfmRecord.load(sfm_json)
    if rec_sfm.model_sha256 != prov.get("sfm_model_sha256"):
        raise ContractError(
            f"{ds}: the recorded SfM model digest does not match the record kept beside it"
        )
    reg_json = prov_dir / "registration.json"
    if manifest.registration is None or not manifest.registration.registration_id:
        raise ContractError(f"{ds}: image-derived dataset with no registration in its manifest")
    if not reg_json.is_file():
        raise ContractError(f"{ds}: no copy of the registration record under {PROVENANCE_DIR}")
    if sha256_file(reg_json) != manifest.registration.record_sha256:
        raise ContractError(
            f"{ds}: the manifest's registration digest does not match the record kept beside "
            "it; the numbers in the manifest are not the ones that were measured"
        )
    rec_reg = RegistrationRecord.load(reg_json)
    # The record must still follow from its own evidence — claim_allowed and the support
    # extent are re-derived, not read (§Phase 3 AD-2, `check_registration`).
    check_registration(rec_reg, rec_sfm.model_sha256)
    if rec_reg.sfm_id != rec_sfm.sfm_id:
        raise ContractError(
            f"{ds}: the registration beside this dataset is of SfM {rec_reg.sfm_id}, not "
            f"{rec_sfm.sfm_id}"
        )
    _check_manifest_matches_registration(ds, manifest, rec_reg)
    _check_init_is_sfm_geometry(ds, manifest, rec_sfm, rec_reg)


def _check_manifest_matches_registration(ds: Path, manifest: Manifest, rec_reg: Any) -> None:
    """The manifest's copy of the registration must be that registration.

    The protocol judge reads ``manifest.registration``, not the record. The digest above proves
    the record is unedited and says nothing about the copy: a manifest with ``claim_allowed``
    flipped to true, or with the support ranges emptied, sits beside a perfectly intact record
    and grants the claim the record refuses. Every field the judge or a report can reach is
    compared, from one list, so a field added to the copy starts being checked rather than
    silently travelling unverified.
    """
    want = claim_bearing_values(rec_reg)
    have = {
        "registration_id": manifest.registration.registration_id,
        "basis": manifest.registration.basis,
        "scale": manifest.registration.scale,
        "support_ranges_m": manifest.registration.support_ranges_m,
        "claim_allowed": manifest.registration.claim_allowed,
        "claim_refusals": list(manifest.registration.claim_refusals),
        "transform": manifest.registration.transform,
        "rmse_m": manifest.registration.rmse_m,
        "inlier_ratio": manifest.registration.inlier_ratio,
        "n_correspondences": manifest.registration.n_correspondences,
        "inlier_threshold_m": manifest.registration.inlier_threshold_m,
        "method": manifest.registration.method,
    }
    differ = [
        f"{k}: manifest {have[k]!r} vs record {want[k]!r}"
        for k in CLAIM_BEARING_FIELDS
        if not _same_value(have[k], want[k])
    ]
    if differ:
        raise ContractError(
            f"{ds}: the manifest's registration block is not the registration it names "
            f"({'; '.join(differ)}). The protocol judge reads the manifest, so a copy that "
            "disagrees with the record grants what the measurement refused."
        )
    if manifest.scale is None or manifest.scale.factor != want["scale"]:
        raise ContractError(
            f"{ds}: scale.factor is {manifest.scale and manifest.scale.factor!r}, but the "
            f"measured scale is {want['scale']!r}"
        )


def _same_value(a: Any, b: Any) -> bool:
    """Compare a manifest copy with a record value, tolerating float round-trips only."""
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-9 * max(1.0, abs(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, tuple) or isinstance(b, tuple):
        return _same_value(list(a) if a is not None else a, list(b) if b is not None else b)
    return a == b


def _init_voxel_m(ds: Path) -> float | None:
    prov = json.loads((ds / PROVENANCE_DIR / "init_provenance.json").read_text())
    v = prov.get("init_voxel_m")
    return float(v) if v else None


def _check_init_is_sfm_geometry(ds: Path, manifest: Manifest, rec_sfm: Any, rec_reg: Any) -> None:
    """Re-derive this dataset's geometry from the reconstruction it names, and compare.

    The earlier version of this check compared ``init_points.ply`` against ``sparse/0`` — two
    clouds inside the same dataset. Replace both with the same TLS-derived geometry and they
    agree with each other perfectly, so the check passed on a dataset whose initialisation
    came from a scanner. Two copies of a claim are not evidence for it.

    So the comparison now starts outside the dataset, from the bundled ``SFM_INTERNAL`` model
    whose digest must still be the one ``SfmRecord`` names, and walks the same path the builder
    walked: the measured Sim(3), then the local origin, then the holdout exclusion. Both of the
    dataset's clouds must be subsets of what comes out. The exact subsample is not re-derived —
    that would pin the check to an RNG call order — but membership is, and a cloud from another
    instrument is metres away, not a different draw.
    """
    from scipy.spatial import cKDTree

    from minegs.core.pointcloud import read_ply

    if rec_sfm.frame != Frame.SFM_INTERNAL.value:  # pragma: no cover - schema pins it
        raise ContractError(f"{ds}: the recorded SfM model is not in SFM_INTERNAL")
    bundled = ds / PROVENANCE_DIR / SFM_MODEL_DIR
    missing = [n for n in SPARSE_FILES if not (bundled / n).is_file()]
    if missing:
        raise ContractError(
            f"{ds}: the reconstruction this dataset is made of is not kept beside it "
            f"({missing} missing from {PROVENANCE_DIR}/{SFM_MODEL_DIR}). Without it, "
            "'the initialisation is this reconstruction's own points' can only be checked "
            "against another copy of this dataset's own geometry."
        )
    digest = model_digest(bundled)
    if digest != rec_sfm.model_sha256:
        raise ContractError(
            f"{ds}: the bundled SfM model hashes to {digest[:12]}, but the record beside it "
            f"names {rec_sfm.model_sha256[:12]}. This is not the reconstruction the dataset "
            "says it was built from."
        )

    # The builder's own path, in the same order and through the same functions.
    model_tls = transform_model(colmap_io.read_model(bundled), rec_reg.sim3())
    origin = np.asarray(manifest.T_tls_from_local.t, dtype=np.float64)
    expected = model_tls.points_xyz() - origin
    holdout = [(float(lo), float(hi)) for lo, hi in _declared_holdout(manifest)]
    if holdout:
        cl = _centerline_of(ds, manifest)
        s = np.atleast_1d(cl.project(expected + origin)[0])
        keep = np.ones(len(expected), bool)
        for lo, hi in holdout:
            keep &= ~((s >= lo) & (s <= hi))
        expected = expected[keep]
    if not len(expected):
        raise ContractError(f"{ds}: the reconstruction has no points left outside the holdout")

    voxel = _init_voxel_m(ds)
    # A voxel downsample moves a point by at most half a voxel diagonal; the floor keeps a fine
    # voxel from making this sensitive to float noise. What it catches is another instrument.
    tol = max(voxel, 0.05) if voxel else 0.05
    tree = cKDTree(expected)
    init = read_ply(ds / manifest.initialization.file)
    sparse = colmap_io.read_model(ds / "sparse" / "0")
    if not sparse.points3D:
        raise ContractError(f"{ds}: sparse/0 holds no points")
    for what, xyz in (
        ("init_points.ply", init.xyz if len(init) <= 20_000 else init.subsample(20_000, 0).xyz),
        ("sparse/0", sparse.points_xyz()),
    ):
        if not len(xyz):
            raise ContractError(f"{ds}: {what} is empty")
        d, _ = tree.query(np.asarray(xyz, dtype=np.float64))
        if float(np.max(d)) > tol:
            raise ContractError(
                f"{ds}: {what} is up to {float(np.max(d)):.3f} m from the registered "
                "reconstruction this dataset names. Its geometry did not come from that "
                "reconstruction, whatever the manifest says."
            )

    # The cameras too. Points alone would leave the poses swappable, and a dataset whose views
    # are somebody else's views renders depth of a tunnel these images never saw. This does not
    # make the golden gate redundant: it catches *substituted* poses, while a reconstruction
    # whose own poses and points disagree reproduces here faithfully and is caught by looking.
    want = {n: c - origin for n, c in _camera_centres(model_tls).items()}
    got = _camera_centres(sparse)
    unknown = sorted(set(got) - set(want))
    if unknown:
        raise ContractError(
            f"{ds}: sparse/0 registers images {unknown[:5]} that the reconstruction beside it "
            "does not. These views came from somewhere else."
        )
    off = {n: float(np.linalg.norm(got[n] - want[n])) for n in got}
    worst = max(off.items(), key=lambda kv: kv[1], default=(None, 0.0))
    if worst[0] is not None and worst[1] > tol:
        raise ContractError(
            f"{ds}: camera {worst[0]!r} sits {worst[1]:.3f} m from where the registered "
            "reconstruction puts it. The dataset's poses are not this reconstruction's poses."
        )


def _declared_holdout(manifest: Manifest) -> list[tuple[float, float]]:
    ho = manifest.split.geometry_holdout
    return [] if ho is None else [tuple(r) for r in ho.chainage_ranges_m]


def _centerline_of(ds: Path, manifest: Manifest) -> Centerline:
    ref = manifest.centerline
    if ref is None:
        raise ContractError(
            f"{ds}: a chainage holdout is declared with no reference axis to measure it along"
        )
    return Centerline.from_csv(ds / ref.file, ref.frame, ref.source)


__all__ = [
    "CENTERLINE_FILE",
    "PROVENANCE_DIR",
    "SfmDatasetConfig",
    "build_dataset_from_sfm",
    "check_image_only_dataset",
    "transform_model",
]
