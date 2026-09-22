"""Evaluation protocol judgement (§5). Leak prevention at the contract level (§1.6).

| protocol          | init                | train images | evaluated on          | claims          |
|-------------------|---------------------|--------------|-----------------------|-----------------|
| reconstruction    | all TLS             | all          | none                  | none            |
| novel_view        | train groups        | train groups | test group images     | render          |
| geometry_holdout  | minus holdout range | per config   | holdout-range TLS     | geometry/volume |
| change            | one of the above per epoch  |      | same range, 2 epochs  | change volume   |

``judge(manifest)`` decides the first three rows. The ``change`` row is a *pair-level*
protocol and is deliberately unreachable from one manifest: ``judge`` never returns
``Protocol.CHANGE`` and never grants ``Claim.CHANGE_VOLUME``. The pair evaluator
(``judge_change(manifest_a, manifest_b)``) is Phase 7 (docs/ROADMAP.md).

``judge(manifest)`` derives which claims a run on this dataset may make. ``require`` raises
``ProtocolViolation`` for anything else — e.g. asking for geometry accuracy from a
``reconstruction`` run.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from minegs.core.errors import ProtocolViolation
from minegs.core.manifest import Manifest


class Protocol(str, Enum):
    RECONSTRUCTION = "reconstruction"
    NOVEL_VIEW = "novel_view"
    GEOMETRY_HOLDOUT = "geometry_holdout"
    CHANGE = "change"  # pair-level; never produced by judge(manifest)


class Claim(str, Enum):
    RENDER_QUALITY = "render_quality"  # PSNR/SSIM/LPIPS on test groups
    GEOMETRY_ACCURACY = "geometry_accuracy"  # accuracy/completeness/chamfer on holdout TLS
    VOLUME_ACCURACY = "volume_accuracy"  # sections / ∫A ds / overbreak on holdout range
    CHANGE_VOLUME = "change_volume"  # epoch differencing — pair protocol only (Phase 7)
    GEOMETRY_DIAGNOSTIC = (
        "geometry_diagnostic"  # numbers allowed, but *not* a claim (fit to train data)
    )


class Judgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocols: list[Protocol]
    claims: list[Claim]
    holdout_ranges_m: list[tuple[float, float]] = Field(default_factory=list)
    test_groups: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    refusals: list[str] = Field(default_factory=list)

    @property
    def primary(self) -> Protocol:
        return self.protocols[0]

    def allows(self, claim: Claim) -> bool:
        return claim in self.claims


def judge(manifest: Manifest) -> Judgement:
    reasons: list[str] = []
    refusals: list[str] = []
    protocols: list[Protocol] = []
    claims: list[Claim] = [Claim.GEOMETRY_DIAGNOSTIC]
    split, init = manifest.split, manifest.initialization

    metric_ok = manifest.scale is not None
    if not metric_ok:
        refusals.append("scale.basis missing -> no metric (geometry/volume) claims (§4)")
    if manifest.source in ("video", "video360"):
        reg = manifest.registration
        if reg is None:
            metric_ok = False
            refusals.append(
                "video source without registration diagnostics -> no metric claims (§7)"
            )
        elif reg.inlier_ratio <= 0 or reg.rmse_m <= 0:
            metric_ok = False
            refusals.append("registration has no usable quality metrics -> no metric claims (§7)")
        else:
            # Phase 3 AD-2. The registration artifact already judged itself against its own
            # quality gate and its own support; what it could not know is this dataset's
            # holdout, so the overlap is decided here.
            if not reg.registration_id:
                metric_ok = False
                refusals.append(
                    "registration block names no registration artifact -> no metric claims: "
                    "numbers with no record behind them are numbers somebody typed (§Phase 3)"
                )
            if reg.claim_allowed is not True:
                metric_ok = False
                why = "; ".join(reg.claim_refusals) or "the registration does not permit a claim"
                refusals.append(f"registration does not permit a metric claim: {why}")

    # ---- novel_view
    if split.test_groups:
        leak = sorted(set(init.groups) & set(split.test_groups))
        if leak:
            refusals.append(f"init uses test groups {leak} -> render claims void")
        else:
            protocols.append(Protocol.NOVEL_VIEW)
            claims.append(Claim.RENDER_QUALITY)
            reasons.append(f"test groups {split.test_groups} are held out from init and training")

    # ---- geometry_holdout
    ho = split.geometry_holdout
    if ho and ho.chainage_ranges_m:
        ok = True
        if not ho.points_excluded:
            ok = False
            refusals.append(
                "geometry_holdout.points_excluded=false -> holdout geometry present in init"
            )
        missing = [
            r
            for r in ho.chainage_ranges_m
            if not any(lo <= r[0] and hi >= r[1] for lo, hi in init.excluded_chainage_ranges_m)
        ]
        if missing:
            # This used to apply only when the init came from TLS, which exempted exactly the
            # image-only case: `sfm_sparse` structure covers the holdout chainage as readily as
            # a scanner does, because the frames that saw it were reconstructed too.
            ok = False
            refusals.append(
                f"holdout ranges {missing} not declared in initialization.excluded_chainage_ranges_m"
            )
        overlap = _registration_overlap(manifest, ho.chainage_ranges_m)
        if overlap:
            ok = False
            refusals.append(
                f"registration support overlaps the evaluation holdout at {overlap} -> no "
                "geometry/volume claim: the transform was fitted against the geometry the "
                "accuracy would be measured on (Phase 3 AD-2). The numbers remain diagnostic."
            )
        if ok and metric_ok:
            protocols.append(Protocol.GEOMETRY_HOLDOUT)
            claims += [Claim.GEOMETRY_ACCURACY, Claim.VOLUME_ACCURACY]
            reasons.append(
                f"holdout chainage {ho.chainage_ranges_m} excluded from init; images_excluded={ho.images_excluded} ({'extrapolation' if ho.images_excluded else 'reconstruction'} test)"
            )
    elif metric_ok:
        reasons.append(
            "no chainage holdout -> geometry numbers are diagnostic only (fit to training TLS)"
        )

    # ---- change: NEVER from a single manifest (§5 "change = same range, 2 epochs").
    # A change claim needs an epoch *pair* contract (different epoch ids, compatible frames and
    # scale basis, common reference axis, overlapping evaluation chainage, leak-free ranges).
    # That is judge_change(manifest_a, manifest_b), Phase 7 — see docs/ROADMAP.md.
    if manifest.capture_epoch is not None:
        # Deliberately not naming the claim token here: a single manifest's judgement must
        # never put it in front of a reader (or a grep) as if it were on offer.
        reasons.append(
            f"capture_epoch {manifest.capture_epoch.id!r} is declared; epoch-difference claims "
            "need a two-epoch pair protocol and are out of scope for a single manifest (Phase 7)"
        )

    if not protocols:
        protocols.append(Protocol.RECONSTRUCTION)
        reasons.append(
            "no held-out groups or ranges -> reconstruction (best quality, no performance claims)"
        )

    return Judgement(
        protocols=protocols,
        claims=claims,
        holdout_ranges_m=list(ho.chainage_ranges_m) if ho else [],
        test_groups=list(split.test_groups),
        reasons=reasons,
        refusals=refusals,
    )


def require(manifest: Manifest, claim: Claim) -> Judgement:
    j = judge(manifest)
    if not j.allows(claim):
        if claim is Claim.CHANGE_VOLUME:
            raise ProtocolViolation(
                "change_volume is a two-epoch claim and can never come from a single manifest "
                f"({manifest.dataset_id!r}). The epoch-pair protocol (judge_change) is Phase 7; "
                "see docs/ROADMAP.md."
            )
        detail = "; ".join(j.refusals) or "manifest declares no split supporting it"
        raise ProtocolViolation(
            f"dataset {manifest.dataset_id!r} (protocol {j.primary.value}) cannot claim {claim.value}: {detail}"
        )
    return j


def _registration_overlap(
    manifest: Manifest, holdout_ranges: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Where the registration support and the evaluation holdout are the same tunnel.

    Support that was never recorded is treated as covering everything, so "we did not write it
    down" refuses rather than passes. A dataset with no registration (the TLS path) has no
    support to conflict with.
    """
    from minegs.eval.register.models import ranges_overlap

    reg = manifest.registration
    if reg is None or not holdout_ranges:
        return []
    if reg.support_ranges_m is None and reg.registration_id is None:
        return []
    return ranges_overlap(reg.support_ranges_m, [tuple(r) for r in holdout_ranges])
