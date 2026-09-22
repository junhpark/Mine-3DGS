"""The registration artifact (§Phase 3 C2) — a measured transform and what measured it.

Registration is where an independent reconstruction stops being unitless. That makes it the
one place where a claim can be corrupted quietly, and the corruption is not a wrong transform:
it is a *right* transform fitted to the same geometry the accuracy will later be measured
against. The numbers come out small because the reconstruction was pulled onto the reference,
and nothing in them says so.

So this record is built around one definition (Phase 3 AD-2):

    registration support = every piece of geometry that moved ``T_tls_from_sfm``
                         = the initial correspondences  ∪  the ICP target

``basis`` says where the *scale* came from. It does not say where the *pose* came from, and a
survey target that fixes the scale does not license refining the pose against the evaluation
cloud. An ICP run against the whole TLS reference therefore stays diagnostic whatever the basis
field says, because no holdout is disjoint from the whole of it.

The record also carries the things the audit found being decided in silence: whether the RANSAC
fit fell back to using every correspondence, whether ICP converged, and which thresholds — if
any — the quality gate was judged against. No thresholds means no claim: a gate with no numbers
in it passes everything, and an invented number would be worse than none.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.frames import Frame, Sim3
from minegs.core.provenance import ProvenanceRecord
from minegs.eval.register.diagnostics import RegistrationDiagnostics

REGISTRATION_FILE = "registration.json"

#: Where the metric scale came from. ``known_target`` is evidence from outside the TLS —
#: survey control, a measured baseline. ``sim3_to_tls`` is the TLS itself.
RegistrationBasis = Literal["known_target", "sim3_to_tls"]

#: What a piece of registration support is. ``tls_whole`` is called out separately because it
#: is the case that can never carry a claim: support that covers everything leaves no holdout
#: outside it.
SupportKind = Literal["targets", "tls_subset", "tls_whole", "none"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SupportRecord(_Strict):
    """One body of geometry the transform was fitted against."""

    kind: SupportKind
    #: File name only — the digest is the identity.
    file: str | None = None
    sha256: str | None = None
    n_points: int | None = None
    #: Chainage the support covers. ``None`` means nobody recorded it, which is not the same
    #: as "none": it makes overlap with the evaluation holdout undecidable, and undecidable is
    #: a refusal.
    ranges_m: list[tuple[float, float]] | None = None
    note: str = ""

    @property
    def is_tls(self) -> bool:
        return self.kind in ("tls_subset", "tls_whole")


class IcpRecord(_Strict):
    used: bool
    converged: bool | None = None
    iterations: int | None = None
    rmse_m: float | None = None
    inlier_ratio: float | None = None
    max_dist_m: float | None = None


class QualityGate(_Strict):
    """What the registration was judged against, and what the judgement was.

    ``thresholds is None`` is the normal state until pilot data fixes them, and it means the
    registration is diagnostic. That is deliberate: a gate with no numbers admits everything,
    and numbers invented here would be a threshold tuned on nothing.
    """

    thresholds: dict[str, float] | None = None
    passed: bool = False
    reasons: list[str] = Field(default_factory=list)


class RegistrationRecord(VersionedModel):
    """``<registration_dir>/registration.json``."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    registration_id: str = Field(min_length=1)
    sfm_id: str = Field(min_length=1)
    sfm_model_sha256: str = Field(min_length=1)
    basis: RegistrationBasis
    source_frame: Literal["SFM_INTERNAL"] = Frame.SFM_INTERNAL.value
    target_frame: Literal["TLS_GLOBAL"] = Frame.TLS_GLOBAL.value
    initial_support: SupportRecord
    icp_support: SupportRecord | None = None
    #: The union of the TLS support above. ``None`` when any contributing support failed to
    #: declare its extent, so that "we cannot tell" is stored as itself.
    support_ranges_m: list[tuple[float, float]] | None = None
    #: The measured Sim(3), as a 4x4. Scale is part of it; that is the whole point.
    T_tls_from_sfm: list[list[float]]
    scale: float = Field(gt=0)
    icp: IcpRecord
    diagnostics: RegistrationDiagnostics
    #: The thresholds this registration was judged against and the verdict. Inside the record
    #: because ``check_registration`` re-derives the claim from it: a verdict kept in a file
    #: beside the record is a verdict nothing can re-check, and the claim rests on it.
    quality_gate: QualityGate
    #: True when the robust fit gave up on its inliers and used every correspondence. The old
    #: code did this silently, which turned a failed robust fit into a confident-looking one.
    ransac_fallback_used: bool = False
    #: The reference cloud the diagnostics were computed against, by digest.
    reference_file: str | None = None
    reference_sha256: str | None = None
    claim_allowed: bool
    claim_refusals: list[str] = Field(default_factory=list)
    provenance: ProvenanceRecord

    def sim3(self) -> Sim3:
        return Sim3.from_matrix(self.T_tls_from_sfm)


def union_ranges(*groups: list[tuple[float, float]] | None) -> list[tuple[float, float]] | None:
    """Merge chainage intervals, or ``None`` if any contributing group is unknown."""
    out: list[tuple[float, float]] = []
    for g in groups:
        if g is None:
            return None
        out.extend((float(lo), float(hi)) for lo, hi in g)
    if not out:
        return []
    out.sort()
    merged = [out[0]]
    for lo, hi in out[1:]:
        last_lo, last_hi = merged[-1]
        if lo <= last_hi:
            merged[-1] = (last_lo, max(last_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def ranges_overlap(
    a: list[tuple[float, float]] | None, b: list[tuple[float, float]] | None
) -> list[tuple[float, float]]:
    """The intervals common to both. An unknown extent on either side is treated as *all*."""
    if a is None or b is None:
        return [(float("-inf"), float("inf"))]
    out = []
    for lo1, hi1 in a:
        for lo2, hi2 in b:
            lo, hi = max(lo1, lo2), min(hi1, hi2)
            if hi > lo:
                out.append((lo, hi))
    return out


def decide_claim(
    *,
    basis: RegistrationBasis,
    initial_support: SupportRecord,
    icp_support: SupportRecord | None,
    icp: IcpRecord,
    gate: QualityGate,
    ransac_fallback_used: bool,
) -> tuple[bool, list[str], list[tuple[float, float]] | None]:
    """Whether this registration may carry a metric claim, why not, and over what support.

    Ordered so the reasons read as a list of everything wrong rather than the first thing.
    """
    refusals: list[str] = []
    if ransac_fallback_used:
        refusals.append(
            "the robust fit found too few inliers and fell back to using every "
            "correspondence, so the transform was fitted to the outliers as well"
        )
    if icp.used and icp.converged is False:
        refusals.append("ICP did not converge; its transform is where the iteration stopped")
    if icp.used and icp_support is None:
        refusals.append(
            "ICP ran but its target geometry was not recorded, so what the pose was fitted "
            "against is unknown and overlap with the evaluation holdout cannot be decided"
        )
    supports = [s for s in (initial_support, icp_support) if s is not None]
    if any(s.kind == "tls_whole" for s in supports):
        refusals.append(
            "the transform was fitted against the whole TLS reference. Scale from an "
            "independent target does not make the pose independent: no holdout is disjoint "
            "from all of it, so these numbers are diagnostic"
        )
    tls_ranges = [s.ranges_m for s in supports if s.is_tls]
    if any(r is None for r in tls_ranges):
        refusals.append(
            "TLS support was used without recording which chainage it covers; overlap with "
            "the evaluation holdout cannot be decided"
        )
    if basis == "sim3_to_tls" and not any(s.is_tls for s in supports):
        refusals.append("basis is sim3_to_tls but no TLS support is recorded")
    if gate.thresholds is None:
        refusals.append(
            "no registration quality thresholds were configured, so the gate judged nothing"
        )
    elif not gate.passed:
        refusals.extend(gate.reasons or ["the registration quality gate did not pass"])
    support = union_ranges(*tls_ranges) if tls_ranges else []
    return (not refusals), refusals, support


def evaluate_gate(
    diagnostics: RegistrationDiagnostics,
    icp: IcpRecord,
    thresholds: dict[str, float] | None,
) -> QualityGate:
    """Judge a registration against explicit thresholds, on more than one number.

    ``rmse_m`` alone says nothing: it is computed over the inliers, so a fit that matched two
    per cent of the cloud beautifully reports a tiny residual. The inlier ratio and the number
    of correspondences are what make it mean something.
    """
    if thresholds is None:
        return QualityGate(thresholds=None, passed=False, reasons=["no thresholds configured"])
    reasons: list[str] = []
    max_rmse = thresholds.get("max_rmse_m")
    if max_rmse is not None and diagnostics.rmse_m > max_rmse:
        reasons.append(f"rmse {diagnostics.rmse_m:.4f} m > {max_rmse} m")
    min_inlier = thresholds.get("min_inlier_ratio")
    if min_inlier is not None and diagnostics.inlier_ratio < min_inlier:
        reasons.append(f"inlier ratio {diagnostics.inlier_ratio:.3f} < {min_inlier}")
    min_n = thresholds.get("min_correspondences")
    if min_n is not None and diagnostics.n_source < min_n:
        reasons.append(f"{diagnostics.n_source} correspondences < {min_n}")
    if icp.used and icp.converged is False:
        reasons.append("ICP did not converge")
    return QualityGate(thresholds=dict(thresholds), passed=not reasons, reasons=reasons)


def find_registration(path: str | Path) -> Path | None:
    p = Path(path)
    if p.is_dir():
        j = p / REGISTRATION_FILE
        return j if j.is_file() else None
    if p.is_file() and p.name == REGISTRATION_FILE:
        return p
    return None


def load_registration(path: str | Path) -> tuple[RegistrationRecord, Path]:
    found = find_registration(path)
    if found is None:
        raise ContractError(f"{path}: not a registration artifact (no {REGISTRATION_FILE})")
    return RegistrationRecord.load(found), found.parent


def check_registration(rec: RegistrationRecord, sfm_model_sha256: str) -> None:
    """The registration must still be of the reconstruction it says it is, and still say it.

    Two separate things, and the second is the one a reader would skip.

    A transform is meaningless apart from the coordinates it transforms: pair it with a
    different model and it maps that model's points somewhere arbitrary, precisely and
    confidently. So the model digest is compared first.

    Then the claim is **re-derived**. ``claim_allowed``, ``claim_refusals`` and
    ``support_ranges_m`` are conclusions — ``decide_claim`` reached them once, from the support
    records, the ICP record, the robust-fit fallback flag and the quality gate. Reading them
    back is reading a conclusion, and a conclusion in a JSON file is four characters away from
    ``true``. Everything they were derived from is in this record, so they are worked out again
    here and compared; a record whose verdict does not follow from its own evidence is refused,
    whatever it says about itself. The same argument applies to the gate: its verdict is
    re-derived from the diagnostics and the thresholds it names.
    """
    if rec.sfm_model_sha256 != sfm_model_sha256:
        raise ContractError(
            f"registration {rec.registration_id} was measured against SfM model "
            f"{rec.sfm_model_sha256[:12]}, but the model now hashes to {sfm_model_sha256[:12]}. "
            "A measured transform belongs to the reconstruction it was measured on."
        )

    gate = evaluate_gate(rec.diagnostics, rec.icp, rec.quality_gate.thresholds)
    if (gate.passed, sorted(gate.reasons)) != (
        rec.quality_gate.passed,
        sorted(rec.quality_gate.reasons),
    ):
        raise ContractError(
            f"registration {rec.registration_id} records a quality gate that does not follow "
            f"from its own diagnostics: recorded passed={rec.quality_gate.passed} "
            f"{rec.quality_gate.reasons}, re-derived passed={gate.passed} {gate.reasons}"
        )

    allowed, refusals, support = decide_claim(
        basis=rec.basis,
        initial_support=rec.initial_support,
        icp_support=rec.icp_support,
        icp=rec.icp,
        gate=gate,
        ransac_fallback_used=rec.ransac_fallback_used,
    )
    if allowed != rec.claim_allowed or sorted(refusals) != sorted(rec.claim_refusals):
        raise ContractError(
            f"registration {rec.registration_id} says claim_allowed={rec.claim_allowed} with "
            f"refusals {rec.claim_refusals}, but its own support, ICP and gate records give "
            f"claim_allowed={allowed} with refusals {refusals}. The verdict was not derived "
            "from the evidence beside it."
        )
    if _ranges_differ(support, rec.support_ranges_m):
        raise ContractError(
            f"registration {rec.registration_id} records support ranges "
            f"{rec.support_ranges_m}, but the union of its TLS support is {support}. The "
            "extent a claim is judged against is not the extent the transform was fitted to."
        )
    if rec.claim_allowed and support is None:
        raise ContractError(
            f"registration {rec.registration_id} allows a claim while its TLS support has no "
            "recorded extent; overlap with an evaluation holdout cannot be decided"
        )


def _ranges_differ(
    a: list[tuple[float, float]] | None, b: list[tuple[float, float]] | None
) -> bool:
    """Compare chainage intervals as numbers. ``None`` ("unknown") equals only ``None``."""
    if (a is None) != (b is None):
        return True
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return True
    return any(
        abs(float(x0) - float(y0)) > 1e-9 or abs(float(x1) - float(y1)) > 1e-9
        for (x0, x1), (y0, y1) in zip(sorted(a), sorted(b), strict=True)
    )


def support_conflict(
    rec: RegistrationRecord, holdout_ranges: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Where the registration support and the evaluation holdout are the same tunnel."""
    if not holdout_ranges:
        return []
    return ranges_overlap(rec.support_ranges_m, [tuple(r) for r in holdout_ranges])


#: Everything a manifest copies out of a registration and a protocol judge then reads. Named
#: once so the copy and the check cannot drift: a field added here is a field the dataset
#: checker starts comparing.
CLAIM_BEARING_FIELDS: tuple[str, ...] = (
    "registration_id",
    "basis",
    "scale",
    "support_ranges_m",
    "claim_allowed",
    "claim_refusals",
    "transform",
    "rmse_m",
    "inlier_ratio",
    "n_correspondences",
    "inlier_threshold_m",
    "method",
)


def claim_bearing_values(rec: RegistrationRecord) -> dict[str, Any]:
    """The record's own values for the fields a manifest copies."""
    return {
        "registration_id": rec.registration_id,
        "basis": rec.basis,
        "scale": float(rec.scale),
        "support_ranges_m": rec.support_ranges_m,
        "claim_allowed": rec.claim_allowed,
        "claim_refusals": list(rec.claim_refusals),
        "transform": rec.T_tls_from_sfm,
        "rmse_m": float(rec.diagnostics.rmse_m),
        "inlier_ratio": float(rec.diagnostics.inlier_ratio),
        "n_correspondences": rec.diagnostics.n_source,
        "inlier_threshold_m": rec.diagnostics.inlier_threshold_m,
        "method": rec.diagnostics.method,
    }


def describe(rec: RegistrationRecord) -> dict[str, Any]:
    """The registration as the protocol judge and the reports want it."""
    return {
        "registration_id": rec.registration_id,
        "basis": rec.basis,
        "scale": rec.scale,
        "rmse_m": rec.diagnostics.rmse_m,
        "inlier_ratio": rec.diagnostics.inlier_ratio,
        "support_ranges_m": rec.support_ranges_m,
        "claim_allowed": rec.claim_allowed,
        "claim_refusals": list(rec.claim_refusals),
    }


__all__ = [
    "CLAIM_BEARING_FIELDS",
    "REGISTRATION_FILE",
    "IcpRecord",
    "QualityGate",
    "RegistrationBasis",
    "RegistrationRecord",
    "SupportKind",
    "SupportRecord",
    "check_registration",
    "claim_bearing_values",
    "decide_claim",
    "describe",
    "evaluate_gate",
    "find_registration",
    "load_registration",
    "ranges_overlap",
    "support_conflict",
    "union_ranges",
]
