"""Measuring ``SFM_INTERNAL → TLS_GLOBAL`` and recording what it was measured against.

The composition is the whole of the arithmetic:

    T_tls_from_sfm  =  Sim3(ICP refinement)  @  Sim3(initial alignment)

and it used to be written the other way round, as ``SE3 @ Sim3``, which Python refuses —
``SE3.__matmul__`` returns ``NotImplemented`` for a Sim3 and Sim3 has no ``__rmatmul__``. Every
invocation of the only registration command in the repository raised ``TypeError``. The fix is
to lift the refinement into a Sim3 and compose in that order, so the scale the initial fit
measured survives a refinement that is rigid by construction.

Everything else here is about what the transform was fitted against, because that is what
decides whether the numbers downstream are a claim or a picture (§Phase 3 AD-2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from minegs.core.errors import ContractError
from minegs.core.frames import SE3, Frame, Sim3
from minegs.core.pointcloud import read_ply
from minegs.core.provenance import make_id, sha256_file, stamp
from minegs.eval.register.diagnostics import diagnose
from minegs.eval.register.initial_alignment import (
    align_correspondences,
    load_targets_csv,
    match_by_id,
)
from minegs.eval.register.models import (
    REGISTRATION_FILE,
    IcpRecord,
    RegistrationBasis,
    RegistrationRecord,
    SupportRecord,
    check_registration,
    decide_claim,
    evaluate_gate,
    union_ranges,
)
from minegs.eval.register.rigid_icp import icp_point_to_point
from minegs.ingest.video.sfm.models import check_sfm, load_sfm


def _sfm_points(model_dir: Path) -> np.ndarray:
    from minegs.ingest.common import colmap_io

    model = colmap_io.read_model(model_dir)
    if not model.points3D:
        raise ContractError(
            f"{model_dir} holds no 3D points. There is nothing to register: a registration is "
            "measured between geometry, not between camera counts."
        )
    return model.points_xyz()


def register_sfm(
    sfm_dir: str | Path,
    out_dir: str | Path,
    *,
    basis: RegistrationBasis,
    targets_sfm: str | Path | None = None,
    targets_tls: str | Path | None = None,
    target_ranges_m: list[tuple[float, float]] | None = None,
    icp_target_ply: str | Path | None = None,
    icp_ranges_m: list[tuple[float, float]] | None = None,
    icp_target_is_whole_reference: bool = False,
    reference_ply: str | Path | None = None,
    icp_max_dist_m: float = 0.5,
    icp_iters: int = 50,
    inlier_m: float | None = None,
    thresholds: dict[str, float] | None = None,
    overwrite: bool = False,
) -> tuple[RegistrationRecord, Path]:
    """Measure the transform, judge it, and write the artifact.

    ``targets_sfm`` / ``targets_tls`` are the correspondences the initial Sim(3) is fitted to.
    ``basis`` says what the TLS side of them is: independent survey control (``known_target``)
    or coordinates read off the reference cloud (``sim3_to_tls``). The difference is not
    cosmetic — it decides whether those points count as registration support.

    There is no path here that assumes the reconstruction is already metric. The old command
    printed a warning and used the identity when no targets were given, which turned "we do not
    know the scale" into "the scale is one".
    """
    rec_sfm, sfm_root = load_sfm(sfm_dir)
    model_dir = check_sfm(rec_sfm, sfm_root)

    if targets_sfm is None or targets_tls is None:
        raise ContractError(
            "registration needs correspondences: --targets (in the reconstruction's own "
            "coordinates) and --targets-tls (in TLS_GLOBAL). Without them there is no measured "
            "scale, and assuming the reconstruction is already metric would invent the one "
            "number this step exists to determine."
        )

    src_map = load_targets_csv(targets_sfm)
    dst_map = load_targets_csv(targets_tls)
    src_pts, dst_pts, ids = match_by_id(src_map, dst_map)
    T0, _inliers, fell_back = align_correspondences(
        src_pts, dst_pts, with_scale=True, inlier_m=inlier_m if inlier_m is not None else 0.1
    )

    initial_support = SupportRecord(
        kind="targets" if basis == "known_target" else "tls_subset",
        file=Path(targets_tls).name,
        sha256=sha256_file(targets_tls),
        n_points=len(ids),
        # Targets that are independent of the TLS contribute no TLS extent, so the holdout
        # stays whole; TLS-read coordinates have to declare the chainage they came from.
        ranges_m=None if basis == "known_target" else target_ranges_m,
        note=(
            "independent survey control; not TLS geometry"
            if basis == "known_target"
            else "coordinates read from the TLS reference"
        ),
    )
    points = _sfm_points(model_dir)
    icp = IcpRecord(used=False)
    icp_support: SupportRecord | None = None
    T = T0

    if icp_target_ply is not None:
        target_cloud = read_ply(icp_target_ply)
        if target_cloud.frame != Frame.TLS_GLOBAL.value:
            raise ContractError(
                f"{icp_target_ply} declares frame {target_cloud.frame}, and ICP refines into "
                "TLS_GLOBAL; a target in another frame would move the reconstruction somewhere "
                "that is not the survey"
            )
        res = icp_point_to_point(
            T0.apply(points),
            target_cloud.xyz,
            SE3.identity(),
            max_dist_m=icp_max_dist_m,
            max_iters=icp_iters,
        )
        # The refinement is rigid and the initial fit carries the scale, so the composition has
        # to happen in Sim(3). This is the line that used to raise TypeError.
        T = Sim3.from_se3(res.T) @ T0
        icp = IcpRecord(
            used=True,
            converged=res.converged,
            iterations=res.iterations,
            rmse_m=res.rmse_m,
            inlier_ratio=res.inlier_ratio,
            max_dist_m=icp_max_dist_m,
        )
        icp_support = SupportRecord(
            kind="tls_whole" if icp_target_is_whole_reference else "tls_subset",
            file=Path(icp_target_ply).name,
            sha256=sha256_file(icp_target_ply),
            n_points=len(target_cloud),
            ranges_m=None if icp_target_is_whole_reference else icp_ranges_m,
            note=(
                "the whole reference cloud: support covers everything, so nothing is held out"
                if icp_target_is_whole_reference
                else "a declared subset of the reference"
            ),
        )

    ref_path = Path(reference_ply) if reference_ply is not None else None
    if ref_path is not None:
        reference = read_ply(ref_path)
    elif icp_target_ply is not None:
        reference, ref_path = read_ply(icp_target_ply), Path(icp_target_ply)
    else:
        reference, ref_path = None, None
    if reference is None:
        raise ContractError(
            "registration diagnostics need a reference cloud to measure residuals against; "
            "pass --reference-ply"
        )

    diagnostics = diagnose(
        T,
        points,
        reference.xyz,
        inlier_m=inlier_m if inlier_m is not None else icp_max_dist_m / 5,
        method="sim3+icp" if icp.used else "sim3",
    )
    # A registration with no inliers produces NaN residuals, and NaN does not survive a JSON
    # round trip: the record was written and could not be read back, so the failure surfaced
    # later as a schema error about a null float instead of here, as what it is.
    unusable = sorted(
        name
        for name in ("rmse_m", "median_m", "p90_m", "inlier_ratio", "scale")
        if not np.isfinite(getattr(diagnostics, name))
    )
    if unusable:
        raise ContractError(
            f"the registration diagnostics are not numbers ({', '.join(unusable)}): the "
            f"transform put {diagnostics.n_source} reconstruction points nowhere near the "
            "reference, so there is nothing to judge. Check the correspondences and the "
            "reference cloud rather than recording this as a measurement."
        )
    gate = evaluate_gate(diagnostics, icp, thresholds)
    allowed, refusals, support = decide_claim(
        basis=basis,
        initial_support=initial_support,
        icp_support=icp_support,
        icp=icp,
        gate=gate,
        ransac_fallback_used=fell_back,
    )

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise ContractError(f"{out} is not empty; a registration artifact needs its own place")
    out.mkdir(parents=True, exist_ok=True)

    options: dict[str, Any] = {
        "basis": basis,
        "icp": icp.used,
        "icp_max_dist_m": icp_max_dist_m,
        "thresholds": thresholds,
    }
    record = RegistrationRecord(
        registration_id=make_id("reg"),
        sfm_id=rec_sfm.sfm_id,
        sfm_model_sha256=rec_sfm.model_sha256,
        basis=basis,
        initial_support=initial_support,
        icp_support=icp_support,
        support_ranges_m=support,
        T_tls_from_sfm=T.to_list(),
        scale=float(T.s),
        icp=icp,
        diagnostics=diagnostics,
        quality_gate=gate,
        ransac_fallback_used=fell_back,
        reference_file=ref_path.name if ref_path else None,
        reference_sha256=sha256_file(ref_path) if ref_path else None,
        claim_allowed=allowed,
        claim_refusals=refusals,
        provenance=stamp(options, parents=[rec_sfm.sfm_id]),
    )
    check_registration(record, rec_sfm.model_sha256)
    # One copy of the verdict, inside the record that rests on it. A second file beside it
    # would be a verdict nothing re-derives and nothing compares.
    (out / REGISTRATION_FILE).write_text(record.model_dump_json(indent=2))
    return record, out


def registered_points(rec: RegistrationRecord, points_sfm: np.ndarray) -> np.ndarray:
    """Reconstruction points in TLS_GLOBAL. The measured Sim(3) applied once, by one function."""
    return rec.sim3().apply(points_sfm)


__all__ = ["register_sfm", "registered_points", "union_ranges"]
