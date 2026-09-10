"""Evaluation protocol judgement (§5). Leak prevention at the contract level (§1.6).

| protocol          | init                | train images | evaluated on          | claims          |
|-------------------|---------------------|--------------|-----------------------|-----------------|
| reconstruction    | all TLS             | all          | none                  | none            |
| novel_view        | train groups        | train groups | test group images     | render          |
| geometry_holdout  | minus holdout range | per config   | holdout-range TLS     | geometry/volume |
| change            | one of the above per epoch  |      | same range, 2 epochs  | change volume   |

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
    CHANGE = "change"


class Claim(str, Enum):
    RENDER_QUALITY = "render_quality"  # PSNR/SSIM/LPIPS on test groups
    GEOMETRY_ACCURACY = "geometry_accuracy"  # accuracy/completeness/chamfer on holdout TLS
    VOLUME_ACCURACY = "volume_accuracy"  # sections / ∫A ds / overbreak on holdout range
    CHANGE_VOLUME = "change_volume"  # epoch differencing
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
        if missing and init.source == "tls":
            ok = False
            refusals.append(
                f"holdout ranges {missing} not declared in initialization.excluded_chainage_ranges_m"
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

    if manifest.capture_epoch is not None and metric_ok and Protocol.GEOMETRY_HOLDOUT in protocols:
        claims.append(Claim.CHANGE_VOLUME)

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
        detail = "; ".join(j.refusals) or "manifest declares no split supporting it"
        raise ProtocolViolation(
            f"dataset {manifest.dataset_id!r} (protocol {j.primary.value}) cannot claim {claim.value}: {detail}"
        )
    return j
