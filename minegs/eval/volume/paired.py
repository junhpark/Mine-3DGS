"""Reconstruction against held-out TLS, on one grid (§Phase 2 G2).

Phase 1C made a *predicted* volume honest: integrated only where sections were observed, with
the gaps reported rather than crossed. It says nothing about whether that volume is right. The
Phase 2 gate asks for section-area error, volume error and valid coverage, and none of those is
a property of one series — they are comparisons against a reference the run never saw.

So the reconstruction and the held-out TLS are cut into sections **on the same grid**: same
reference axis, same chainages, same interval, slab and bin count, same holdout ranges. Anything
else and the two areas at "station 22 m" are areas of different things.

The comparison then happens on the **common domain** and nowhere else. A station is paired only
when both sides observed it; a span is integrated only when both sides observed both of its
ends. This is the rule the whole module exists for:

    predicted 100 m³ over one coverage minus reference 102 m³ over another is not a 2 m³ error.

It is two numbers about two different pieces of tunnel. Subtracting them produces something that
looks like an error, reads like an error, and is not one — and it gets smaller exactly when the
reconstruction covers less, which is the wrong direction for a number people will trust.

Nothing here integrates: the runs of paired stations come from ``finite_runs`` and the areas
under them from ``trapezoid``, the same helpers ``integrate_sections`` uses. A second trapezoid
implementation is a second answer waiting to disagree with the first.

This is Phase 2 G2 validation evidence. It is not a new ``Claim``, and it does not upgrade one.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.errors import ContractError
from minegs.eval.volume.coverage import (
    EPS_M,
    Interval,
    finite_runs,
    merge_intervals,
    subtract_intervals,
    total_length,
    trapezoid,
)

__all__ = [
    "PairedSectionReport",
    "PairedStation",
    "PairedValidation",
    "PairedVolumeReport",
    "compare_to_reference",
    "require_same_grid",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PairedStation(_Strict):
    """One chainage, both sides. ``None`` on a side means that side did not observe it."""

    chainage_m: float
    pred_area_m2: float | None = None
    ref_area_m2: float | None = None
    signed_error_m2: float | None = None
    absolute_error_m2: float | None = None
    relative_error: float | None = None
    #: Why ``relative_error`` is null, when it is. A zero reference area is a real thing to
    #: report; dividing by it and shipping ``inf`` is not.
    relative_error_reason: str | None = None


class PairedSectionReport(_Strict):
    requested_intervals_m: list[Interval] = Field(default_factory=list)
    interval_m: float
    thickness_m: float
    angle_bins: int
    station_count: int = 0
    paired_valid_count: int = 0
    missing_prediction_count: int = 0
    missing_reference_count: int = 0
    mean_absolute_error_m2: float | None = None
    median_absolute_error_m2: float | None = None
    p95_absolute_error_m2: float | None = None
    mean_signed_error_m2: float | None = None
    #: Mean over the paired stations whose reference area allows a ratio at all.
    mean_relative_error: float | None = None
    relative_error_station_count: int = 0
    stations: list[PairedStation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PairedVolumeReport(_Strict):
    """Both volumes over the *same* spans, and the spans themselves."""

    predicted_volume_m3: float | None = None
    reference_volume_m3: float | None = None
    signed_error_m3: float | None = None
    absolute_error_m3: float | None = None
    relative_error: float | None = None
    relative_error_reason: str | None = None
    requested_intervals_m: list[Interval] = Field(default_factory=list)
    requested_length_m: float = 0.0
    common_covered_length_m: float = 0.0
    coverage_fraction: float = 0.0
    integrated_intervals_m: list[Interval] = Field(default_factory=list)
    missing_intervals_m: list[Interval] = Field(default_factory=list)
    #: Where only one side observed. Reported separately from "neither did", because the two
    #: say different things about the reconstruction.
    prediction_only_intervals_m: list[Interval] = Field(default_factory=list)
    reference_only_intervals_m: list[Interval] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PairedValidation(_Strict):
    """What the Phase 2 gate asks for, and the grid it was measured on."""

    grid: dict = Field(default_factory=dict)
    sections: PairedSectionReport
    volume: PairedVolumeReport
    notes: list[str] = Field(default_factory=list)


def require_same_grid(pred, ref) -> dict:
    """The two series must be the same question asked of two clouds.

    Chainage is only a shared coordinate if both sides were cut along the same polyline at the
    same stations; area is only comparable if the slab and the bin count match, because both
    change what "the area at 22 m" means. Returns the grid, so the report can state it.
    """
    for name, a, b in (
        ("reference_axis", pred.reference_axis, ref.reference_axis),
        ("dataset_id", pred.dataset_id, ref.dataset_id),
        ("dataset_hash", pred.dataset_hash, ref.dataset_hash),
        ("frame", pred.frame, ref.frame),
        ("interval_m", pred.series.interval_m, ref.series.interval_m),
        ("thickness_m", pred.series.thickness_m, ref.series.thickness_m),
        ("angle_bins", pred.series.angle_bins, ref.series.angle_bins),
    ):
        if a != b:
            raise ContractError(
                f"the prediction and the TLS reference disagree about {name} ({a!r} vs {b!r}); "
                "they were not cut on the same grid, so their areas are areas of different things"
            )
    ps = np.asarray(pred.series.chainages(), dtype=np.float64)
    rs = np.asarray(ref.series.chainages(), dtype=np.float64)
    if len(ps) != len(rs) or (len(ps) and float(np.max(np.abs(np.sort(ps) - np.sort(rs)))) > EPS_M):
        raise ContractError(
            f"the prediction has {len(ps)} stations and the TLS reference has {len(rs)}, or they "
            "sit at different chainages; a paired comparison needs one station list"
        )
    return {
        "reference_axis": pred.reference_axis,
        "interval_m": pred.series.interval_m,
        "thickness_m": pred.series.thickness_m,
        "angle_bins": pred.series.angle_bins,
        "frame": pred.frame,
        "station_count": len(ps),
    }


def _sorted_areas(record) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(record.series.chainages(), dtype=np.float64)
    a = np.asarray(record.series.areas(), dtype=np.float64)
    order = np.argsort(s, kind="stable")
    s, a = s[order], a[order]
    return s, np.where(np.isfinite(a), a, np.nan)


def _inside(s: np.ndarray, intervals: list[Interval]) -> np.ndarray:
    keep = np.zeros(len(s), bool)
    for lo, hi in intervals:
        keep |= (s >= lo - EPS_M) & (s <= hi + EPS_M)
    return keep


def _spans(s: np.ndarray, mask: np.ndarray, step: float) -> list[Interval]:
    """The chainage spans a side covers on its own: runs of ≥ 2 of its observed stations."""
    values = np.where(mask, 0.0, np.nan)
    out = []
    for run in finite_runs(values, at=s, max_step=step):
        if len(run) >= 2:
            out.append((float(s[run[0]]), float(s[run[-1]])))
    return merge_intervals(out)


def compare_to_reference(pred, ref, ranges: list[Interval] | None = None) -> PairedValidation:
    """Section-area and volume error of *pred* against held-out *ref*.

    *ranges* is the declared geometry holdout on the claim path — what the comparison is a
    comparison *about*. Coverage is measured against it, so a reconstruction that reaches half
    of the holdout reports half the coverage rather than full coverage of a shorter tunnel.
    """
    grid = require_same_grid(pred, ref)
    step = float(pred.series.interval_m or 0.0)
    ps, pa = _sorted_areas(pred)
    _rs, ra = _sorted_areas(ref)
    if ranges is None:
        requested = merge_intervals([(float(ps[0]), float(ps[-1]))]) if len(ps) >= 2 else []
    else:
        bad = [(lo, hi) for lo, hi in ranges if not (float(hi) - float(lo) > EPS_M)]
        if bad:
            raise ContractError(f"comparison range must be [lo, hi] with hi > lo, got {bad}")
        requested = merge_intervals(ranges)

    inside = _inside(ps, requested)
    p_ok, r_ok = np.isfinite(pa) & inside, np.isfinite(ra) & inside
    both = p_ok & r_ok

    stations = [
        _station(float(ps[i]), pa[i] if p_ok[i] else None, ra[i] if r_ok[i] else None)
        for i in np.flatnonzero(inside)
    ]
    errs = np.array(
        [st.absolute_error_m2 for st in stations if st.absolute_error_m2 is not None],
        dtype=np.float64,
    )
    signed = np.array(
        [st.signed_error_m2 for st in stations if st.signed_error_m2 is not None],
        dtype=np.float64,
    )
    rels = np.array(
        [st.relative_error for st in stations if st.relative_error is not None], dtype=np.float64
    )
    section_report = PairedSectionReport(
        requested_intervals_m=requested,
        interval_m=pred.series.interval_m,
        thickness_m=pred.series.thickness_m,
        angle_bins=pred.series.angle_bins,
        station_count=int(inside.sum()),
        paired_valid_count=int(both.sum()),
        missing_prediction_count=int((inside & ~p_ok).sum()),
        missing_reference_count=int((inside & ~r_ok).sum()),
        mean_absolute_error_m2=float(errs.mean()) if len(errs) else None,
        median_absolute_error_m2=float(np.median(errs)) if len(errs) else None,
        p95_absolute_error_m2=float(np.percentile(errs, 95)) if len(errs) else None,
        mean_signed_error_m2=float(signed.mean()) if len(signed) else None,
        mean_relative_error=float(rels.mean()) if len(rels) else None,
        relative_error_station_count=len(rels),
        stations=stations,
        notes=_section_notes(int(both.sum()), int(inside.sum())),
    )
    volume_report = _volume(ps, pa, ra, both, p_ok, r_ok, requested, step)
    return PairedValidation(grid=grid, sections=section_report, volume=volume_report)


def _station(chainage: float, pred: float | None, ref: float | None) -> PairedStation:
    st = PairedStation(chainage_m=chainage, pred_area_m2=pred, ref_area_m2=ref)
    if pred is None or ref is None:
        return st
    st.signed_error_m2 = float(pred - ref)
    st.absolute_error_m2 = abs(st.signed_error_m2)
    if abs(ref) <= EPS_M:
        # A reference area of zero is a real observation to report, and a ratio against it is
        # not a number. Saying so beats shipping inf, and beats dropping the station silently.
        st.relative_error_reason = "the reference area is zero, so a ratio is undefined"
        return st
    st.relative_error = st.absolute_error_m2 / abs(float(ref))
    return st


def _section_notes(paired: int, inside: int) -> list[str]:
    if inside == 0:
        return ["no stations fall inside the requested ranges"]
    if paired == 0:
        return ["no station was observed by both sides, so no area error could be computed"]
    return []


def _volume(
    s: np.ndarray,
    pa: np.ndarray,
    ra: np.ndarray,
    both: np.ndarray,
    p_ok: np.ndarray,
    r_ok: np.ndarray,
    requested: list[Interval],
    step: float,
) -> PairedVolumeReport:
    """Both volumes over the runs where *both* sides observed, and nowhere else."""
    paired_values = np.where(both, 0.0, np.nan)
    runs = [r for r in finite_runs(paired_values, at=s, max_step=step) if len(r) >= 2]
    integrated = merge_intervals([(float(s[r[0]]), float(s[r[-1]])) for r in runs])
    missing = subtract_intervals(requested, integrated)
    req_len, cov_len = total_length(requested), total_length(integrated)

    pred_v = sum(trapezoid(pa[r], s[r]) for r in runs) if runs else None
    ref_v = sum(trapezoid(ra[r], s[r]) for r in runs) if runs else None
    report = PairedVolumeReport(
        predicted_volume_m3=pred_v,
        reference_volume_m3=ref_v,
        requested_intervals_m=requested,
        requested_length_m=req_len,
        common_covered_length_m=cov_len,
        coverage_fraction=(cov_len / req_len) if req_len > EPS_M else 0.0,
        integrated_intervals_m=integrated,
        missing_intervals_m=missing,
        prediction_only_intervals_m=subtract_intervals(_spans(s, p_ok, step), integrated),
        reference_only_intervals_m=subtract_intervals(_spans(s, r_ok, step), integrated),
    )
    if pred_v is None or ref_v is None:
        report.notes.append(
            "no span was observed by both sides, so there is nothing the two volumes could be "
            "compared over"
        )
        return report
    report.signed_error_m3 = float(pred_v - ref_v)
    report.absolute_error_m3 = abs(report.signed_error_m3)
    if abs(ref_v) <= EPS_M:
        report.relative_error_reason = "the reference volume is zero, so a ratio is undefined"
    else:
        report.relative_error = report.absolute_error_m3 / abs(ref_v)
    if report.missing_intervals_m:
        report.notes.append(
            "the error above is over the common domain only; "
            f"{req_len - cov_len:.2f} m of the requested span was not observed by both sides"
        )
    return report
