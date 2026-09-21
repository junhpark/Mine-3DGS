"""The geometry claim gate, as a function (§11, §5).

It lived inside the ``minegs eval geometry`` command, which was fine while the CLI was the only
caller. Phase 2 has a second one, and a gate with two implementations is a gate with two
answers: the orchestrated path would drift from the hand-run one exactly where it matters most.
So the decision moved here and the command became a thin caller of it.

One behavioural change came with the move, and it is an improvement rather than a side effect.
The warnings this gate emits — a raw PLY accepted as diagnostic, a reference cloud in the wrong
frame, ``--no-holdout-only`` measuring the training chainage — used to be console lines that
never reached ``geometry.json``. They are returned now, so the caller can print them *and* a
report can carry them. A downgrade nobody can see afterwards is a downgrade that did not
really happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from minegs.core.errors import ContractError, ProtocolViolation
from minegs.core.pointcloud import PointCloud, read_ply
from minegs.eval.geometry.metrics import GeometryReport, compare_clouds
from minegs.eval.protocol import Claim, judge

__all__ = [
    "SURFACE_REQUIRED",
    "GeometryEvaluation",
    "evaluate_geometry",
    "load_dataset_and_centerline",
    "resolve_prediction",
    "to_tls",
    "unverified_depth_message",
]

SURFACE_REQUIRED = (
    "Geometry accuracy requires a surface artifact. Gaussian centres are not surfaces. "
    "Build one with `minegs eval surface-depth <depth_dir> <dataset_dir> --run-dir <run_dir> "
    "--out <surface_dir>` and pass <surface_dir> here, or pass --diagnostic for non-claim "
    "numbers from a raw PLY."
)


def unverified_depth_message(rec) -> str:
    return (
        f"surface {rec.surface_id} was built from depth maps minegs did not render "
        f"(depth_source={rec.depth_source}), so it cannot carry a geometry_accuracy claim. "
        f"Nothing ties those maps to run {rec.run_id}: the same directory paired with any "
        "succeeded run of this dataset would produce the same artifact. Rendering metric depth "
        "from a trained run is Phase 1B (docs/ROADMAP.md). Pass --diagnostic for non-claim "
        "numbers."
    )


def load_dataset_and_centerline(dataset_dir: str | Path):
    from minegs.core.centerline import Centerline
    from minegs.core.manifest import Manifest

    dataset_dir = Path(dataset_dir)
    m = Manifest.load_dataset(dataset_dir, strict_layout=False)
    if m.centerline is None:
        raise ContractError("dataset has no centerline (needed for sections/volume/holdout ranges)")
    cl = Centerline.from_csv(
        dataset_dir / m.centerline.file, m.centerline.frame, m.centerline.source
    )
    if m.centerline.frame == "LOCAL_METRIC":
        cl = cl.transformed(m.T_tls_from_local, "TLS_GLOBAL")
    return m, cl


def to_tls(pc: PointCloud, m, notes: list[str] | None = None) -> PointCloud:
    if pc.frame == "TLS_GLOBAL":
        return pc
    if pc.frame in ("LOCAL_METRIC", "UNKNOWN"):
        if pc.frame == "UNKNOWN" and notes is not None:
            notes.append("PLY has no frame comment; assuming LOCAL_METRIC")
        return pc.transformed(m.T_tls_from_local, "TLS_GLOBAL")
    raise ContractError(
        f"cannot evaluate a cloud in frame {pc.frame}; adapters must export LOCAL_METRIC (§3)"
    )


@dataclass
class Resolved:
    points: PointCloud
    surface: object | None
    notes: list[str] = field(default_factory=list)


def resolve_prediction(pred: Path, dataset_dir: Path, m, diagnostic: bool) -> Resolved:
    """Resolve the geometry input to points and a ``SurfaceRecord``, or refuse.

    A surface artifact is what separates "these points came off a reconstruction" from "these
    points are where the optimiser put its Gaussians" (§16) — and ``check_surface`` verifies the
    points against the record rather than taking the record's word for them, so wrapping a
    Gaussian PLY in a hand-written surface.json does not get past this. Being a surface is still
    not sufficient: a claim also needs depth this project rendered, and that verdict is
    re-derived rather than read (§1A, §1C).
    """
    from minegs.core.provenance import sha256_tree
    from minegs.eval.surface.depth import rederive_depth_source
    from minegs.eval.surface.models import check_surface, find_surface, load_surface
    from minegs.train.runner.base import DATASET_HASH_PATTERNS

    notes: list[str] = []
    if find_surface(pred) is not None:
        rec, points = load_surface(pred)
        pc = check_surface(
            rec, points, m.dataset_id, sha256_tree(dataset_dir, DATASET_HASH_PATTERNS)
        )
        notes.append(
            f"surface {rec.surface_id} ({rec.method}, depth {rec.depth_source}, "
            f"{rec.point_count} points from {rec.depth_map_count} depth maps, run {rec.run_id})"
        )
        if rec.supports_accuracy_claim:
            # `depth_source` is derived from evidence once, when the surface is fused, and read
            # back forever. check_surface re-reads the points and checks their digest; it has
            # never re-checked this field, so a hand-written surface.json over any PLY --
            # `raw/tls_full.ply` itself, in the audit that found this -- claimed geometry
            # accuracy of 0.0 mm against the cloud it had been copied from. So the value is
            # re-derived here rather than read, and what comes back is what is used.
            stale = rederive_depth_source(rec, dataset_dir)
            if stale is not None:
                if not diagnostic:
                    raise ContractError(stale)
                notes.append(f"warning: {stale}")
                rec.depth_source = "external_unverified"
        if not rec.supports_accuracy_claim:
            if not diagnostic:
                raise ContractError(unverified_depth_message(rec))
            notes.append(
                "warning: this surface's depth maps are unverified external input; these "
                "numbers are diagnostic, not a validated geometry claim"
            )
        return Resolved(pc, rec, notes)
    if not diagnostic:
        raise ContractError(SURFACE_REQUIRED)
    if not pred.is_file():
        raise ContractError(f"{pred}: not a surface artifact (no surface.json) and not a PLY file")
    notes.append(
        "warning: raw PLY accepted only as diagnostic; this is not a validated surface artifact"
    )
    return Resolved(read_ply(pred), None, notes)


@dataclass
class GeometryEvaluation:
    report: GeometryReport
    notes: list[str] = field(default_factory=list)


def evaluate_geometry(
    pred: str | Path,
    dataset_dir: str | Path,
    tls_ply: str | Path,
    holdout_only: bool = True,
    max_dist_m: float = 1.0,
    diagnostic: bool = False,
) -> GeometryEvaluation:
    """Bidirectional accuracy / completeness / Chamfer against TLS, with the claim decided."""
    pred, dataset_dir, tls_ply = Path(pred), Path(dataset_dir), Path(tls_ply)
    m, cl = load_dataset_and_centerline(dataset_dir)
    j = judge(m)
    claim = Claim.GEOMETRY_ACCURACY
    notes: list[str] = []
    if not j.allows(claim):
        if not diagnostic:
            raise ProtocolViolation(
                f"{m.dataset_id}: protocol {j.primary.value} cannot claim geometry_accuracy "
                f"({'; '.join(j.refusals) or 'no holdout'}); pass --diagnostic for non-claim "
                "numbers"
            )
        claim = Claim.GEOMETRY_DIAGNOSTIC
    resolved = resolve_prediction(pred, dataset_dir, m, diagnostic)
    notes += resolved.notes
    surface = resolved.surface
    if surface is None or not surface.supports_accuracy_claim:
        # Neither a raw PLY nor a surface fused from external depth carries an accuracy claim,
        # whatever the manifest would allow: in the first case nothing establishes that these
        # points sample the tunnel wall, in the second nothing ties the depth to the run
        # (§18, §1A). resolve_prediction has already refused unless --diagnostic.
        claim = Claim.GEOMETRY_DIAGNOSTIC
    if claim is Claim.GEOMETRY_ACCURACY and not (holdout_only and j.holdout_ranges_m):
        # geometry_accuracy is defined as accuracy *on holdout TLS* (Claim docstring), and the
        # manifest only grants it because those ranges were excluded from initialisation.
        # Measuring over the whole cloud instead measures the model against the geometry it was
        # initialised and trained on, which is the leak the protocol gate exists to prevent.
        notes.append(
            "warning: --no-holdout-only measures the training chainage too, so these numbers "
            "are fit to the data the run saw; reporting as diagnostic"
        )
        claim = Claim.GEOMETRY_DIAGNOSTIC

    pred_pc = to_tls(resolved.points, m, notes)
    ref = read_ply(tls_ply)
    if ref.frame != "TLS_GLOBAL":
        # The predicted side is refused for exactly this in to_tls; the reference side used to
        # get a console line that never reached the JSON. Comparing a LOCAL_METRIC cloud against
        # a TLS_GLOBAL one measures the distance between two coordinate systems, and
        # `dataset/init_points.ply` sits one directory from `raw/tls_full.ply`, so the mix-up is
        # an ordinary one. UNKNOWN counts as wrong: a cloud that never declared its frame has
        # not been established to be in this one.
        wrong_frame = (
            f"{tls_ply}: reference cloud is in frame {ref.frame}, not TLS_GLOBAL. Geometry is "
            "compared in TLS_GLOBAL, so this would measure the offset between two coordinate "
            "systems rather than between two surfaces. Re-export it with its frame declared, or "
            "pass --diagnostic for non-claim numbers."
        )
        if not diagnostic:
            raise ContractError(wrong_frame)
        notes.append(wrong_frame)

    pxyz, rxyz = pred_pc.xyz, ref.xyz
    rng = None
    if holdout_only and j.holdout_ranges_m:
        sp, _ = cl.project(pxyz)
        sr, _ = cl.project(rxyz)
        mp = np.zeros(len(sp), bool)
        mr = np.zeros(len(sr), bool)
        for lo, hi in j.holdout_ranges_m:
            mp |= (sp >= lo) & (sp <= hi)
            mr |= (sr >= lo) & (sr <= hi)
        pxyz, rxyz = pxyz[mp], rxyz[mr]
        rng = (min(r[0] for r in j.holdout_ranges_m), max(r[1] for r in j.holdout_ranges_m))
    # An accuracy/completeness pair over an empty cloud is not a number, it is a missing input:
    # the comparison would come back all-NaN (or, until this check existed, as a bare KeyError
    # from compare_clouds with no message at all). Name which side emptied and what emptied it,
    # because the usual cause is a reference that does not cover the holdout chainage -- a
    # different TLS epoch, or a different chainage origin.
    empty = [n for n, a in (("prediction", pxyz), ("TLS reference", rxyz)) if len(a) == 0]
    if empty:
        where = f" inside the holdout chainage {rng[0]}-{rng[1]} m" if rng else ""
        raise ContractError(
            f"nothing to compare: the {' and the '.join(empty)} has no points{where}. "
            f"Check that {tls_ply} covers this dataset's chainage and shares its origin."
        )
    rep = compare_clouds(pxyz, rxyz, max_dist_m)
    rep.chainage_range_m = rng
    rep.claim = claim.value
    return GeometryEvaluation(rep, notes)
