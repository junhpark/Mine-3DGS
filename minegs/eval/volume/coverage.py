"""Integrating A(s) without crossing the gaps in it (§11, Phase 1C).

``integrate_sections`` used to drop the invalid stations and hand what was left to
``np.trapezoid``. With areas ``10, 10, -, 10, 10`` at ``s = 0..4`` that is one trapezoid from
1 m to 3 m: 20 m³ of tunnel nobody measured, reported as measured, with the two missing
stations listed underneath as a footnote. The footnote was the whole defence.

So integration happens over *contiguous* runs of observed stations and nowhere else. The
missing span is reported as an interval, not imputed:

``V = Σ segment volumes``, ``missing = requested − Σ segment spans``

Two consequences worth stating, because they are the point rather than a side effect:

* A gap costs its own length of volume, so the number is an **under**-estimate of the real
  excavation whenever coverage is incomplete. That is the safe direction, and
  ``coverage_fraction`` is what tells a reader how far from complete it is.
* Coverage is measured against what was *requested* — the holdout ranges on a claim path — not
  against what happens to be in the series. A section series that stops halfway through the
  holdout has half the coverage, not full coverage of a shorter tunnel.

No threshold lives here. Whether incomplete coverage may still carry a claim is the caller's
decision (``minegs/cli/eval_cmd.py`` refuses it outright for ``volume_accuracy``); this module
only reports what was and was not integrated.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.errors import ContractError

__all__ = [
    "CoverageReport",
    "IntegrationSegment",
    "Interval",
    "finite_runs",
    "integration_segments",
    "merge_intervals",
    "plan_integration",
    "segmented_integral",
    "subtract_intervals",
    "summarise_coverage",
    "trapezoid",
]

Interval = tuple[float, float]

#: Slack for comparing chainages that arithmetic should have made equal — ``np.arange`` steps
#: and a holdout bound written as ``26.0``. One nanometre: it absorbs float representation and
#: nothing else. It is deliberately *not* a coverage tolerance; a millimetre of genuinely
#: unobserved tunnel is still unobserved, and the project does not invent thresholds.
EPS_M = 1e-9


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(y, x))


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Sorted, non-overlapping union. Touching intervals merge; empty ones are dropped."""
    kept = sorted((float(lo), float(hi)) for lo, hi in intervals if float(hi) - float(lo) > EPS_M)
    out: list[Interval] = []
    for lo, hi in kept:
        if out and lo <= out[-1][1] + EPS_M:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def subtract_intervals(base: list[Interval], cut: list[Interval]) -> list[Interval]:
    """``base`` minus ``cut``, both merged first. What is left is what nothing covered."""
    out: list[Interval] = []
    cuts = merge_intervals(cut)
    for lo, hi in merge_intervals(base):
        cursor = lo
        for clo, chi in cuts:
            if chi <= cursor + EPS_M or clo >= hi - EPS_M:
                continue
            if clo > cursor + EPS_M:
                out.append((cursor, min(clo, hi)))
            cursor = max(cursor, chi)
            if cursor >= hi - EPS_M:
                break
        if hi - cursor > EPS_M:
            out.append((cursor, hi))
    return out


def total_length(intervals: list[Interval]) -> float:
    return float(sum(hi - lo for lo, hi in intervals))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntegrationSegment(_Strict):
    """One contiguous run of observed stations, and the volume it alone accounts for."""

    start_chainage_m: float
    end_chainage_m: float
    section_count: int
    volume_m3: float
    mean_area_m2: float


class CoverageReport(_Strict):
    """What was asked for, what was integrated, and what is simply not there.

    ``integrated_intervals_m`` and ``missing_intervals_m`` partition ``requested_intervals_m``
    exactly — that is the invariant that makes this a report rather than a reassurance.
    """

    requested_intervals_m: list[Interval] = Field(default_factory=list)
    integrated_intervals_m: list[Interval] = Field(default_factory=list)
    missing_intervals_m: list[Interval] = Field(default_factory=list)
    requested_length_m: float = 0.0
    covered_length_m: float = 0.0
    coverage_fraction: float = 0.0
    valid_section_count: int = 0
    missing_section_count: int = 0
    #: Stations inside the requested span whose section is invalid. Kept alongside the intervals
    #: because a reader chasing one bad station wants the station, not the span it sits in.
    missing_chainages_m: list[float] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Something was asked for, and all of it was integrated.

        The first half matters: with nothing requested there is nothing missing either, and
        "complete coverage of no tunnel" would read as a satisfied condition on a claim path.
        """
        return bool(self.requested_intervals_m) and not self.missing_intervals_m

    def describe_gaps(self) -> str:
        return ", ".join(f"{lo:g}-{hi:g} m" for lo, hi in self.missing_intervals_m) or "none"


def _stations(series) -> tuple[np.ndarray, np.ndarray]:
    """Chainages and areas, ascending, with a NaN area wherever a section is not usable."""
    s = np.asarray(series.chainages(), dtype=np.float64)
    a = np.asarray(series.areas(), dtype=np.float64)
    if len(s) != len(a):  # pragma: no cover - SectionSeries builds them together
        raise ContractError("section series has a different number of chainages and areas")
    order = np.argsort(s, kind="stable")
    s, a = s[order], a[order]
    if len(s) > 1:
        same = np.diff(s) <= EPS_M
        if same.any():
            where = float(s[:-1][same][0])
            raise ContractError(
                f"two sections share chainage {where:g} m; a series with duplicate stations "
                "integrates the same tunnel twice"
            )
    # An area that is not a finite number is not an observation, whatever `valid` says. Inf
    # would propagate through the trapezoid into the total and out into the report.
    a = np.where(np.isfinite(a), a, np.nan)
    return s, a


def _requested(s: np.ndarray, ranges: list[Interval] | None) -> list[Interval]:
    """The spans a caller asked about: explicit ranges, or the series' own extent."""
    if ranges is None:
        return _series_span(s)
    bad = [(lo, hi) for lo, hi in ranges if not (float(hi) - float(lo) > EPS_M)]
    if bad:
        # merge_intervals would drop these, and a dropped range asks for nothing, which then
        # reports as nothing missing. A malformed span is a caller error, not empty coverage.
        raise ContractError(f"integration range must be [lo, hi] with hi > lo, got {bad}")
    return merge_intervals(ranges)


def plan_integration(
    series, ranges: list[Interval] | None = None
) -> tuple[list[IntegrationSegment], CoverageReport]:
    """What would be integrated, and what would be left out — without integrating anything.

    The claim gate needs the coverage verdict *before* it decides whether a volume may be
    computed at all, and a series with no two consecutive observations in the holdout must
    reach that verdict rather than a bare "nothing to integrate".
    """
    segments = integration_segments(series, ranges)
    return segments, summarise_coverage(series, segments, ranges)


def integration_segments(series, ranges: list[Interval] | None = None) -> list[IntegrationSegment]:
    """The contiguous runs of observed stations that may be integrated.

    *ranges* restricts the integration — the declared geometry holdout, on a claim path. Each
    merged range is integrated independently: two disjoint holdout ranges are two spans of
    tunnel with unmeasured drift in between, and joining them would integrate straight across
    it, which is the same mistake as integrating across a gap.
    """
    s, a = _stations(series)
    out: list[IntegrationSegment] = []
    for lo, hi in _requested(s, ranges):
        inside = np.flatnonzero((s >= lo - EPS_M) & (s <= hi + EPS_M))
        out += _segments_within(s, a, inside)
    return out


def _series_span(s: np.ndarray) -> list[Interval]:
    return [(float(s[0]), float(s[-1]))] if len(s) >= 2 else []


def finite_runs(values: np.ndarray, order: np.ndarray | None = None) -> list[np.ndarray]:
    """Maximal runs of consecutive entries of *order* whose value is finite.

    *order* defaults to every index in turn. This is the one place a gap is recognised, so
    everything that integrates along chainage — area, over/underbreak, epoch difference — cuts
    at exactly the same stations.
    """
    idx = np.arange(len(values)) if order is None else np.asarray(order, dtype=int)
    out: list[np.ndarray] = []
    run: list[int] = []
    for i in [*list(idx), None]:
        if i is not None and np.isfinite(values[i]):
            run.append(int(i))
            continue
        if run:
            out.append(np.array(run, dtype=int))
        run = []
    return out


def segmented_integral(s: np.ndarray, y: np.ndarray) -> tuple[float, list[Interval]]:
    """``Σ ∫y ds`` over the runs of finite *y*, plus the chainage intervals those runs span.

    A run of one station spans no length and contributes nothing — it is an observation with
    no neighbour to integrate towards, and inventing the half-interval around it would be the
    imputation this module exists to refuse.
    """
    total, spans = 0.0, []
    for run in finite_runs(y):
        if len(run) < 2:
            continue
        total += trapezoid(y[run], s[run])
        spans.append((float(s[run[0]]), float(s[run[-1]])))
    return total, spans


def _segments_within(s: np.ndarray, a: np.ndarray, inside: np.ndarray) -> list[IntegrationSegment]:
    """Cut ``inside`` (indices into *s*, ascending and contiguous) at every unobserved station."""
    out: list[IntegrationSegment] = []
    for idx in finite_runs(a, inside):
        # A run of one station spans no length, so it integrates to nothing. It is still an
        # observation, and the span around it is still missing: both are true, and reporting a
        # segment of zero length would only put a 0 m³ row in front of the reader.
        if len(idx) < 2:
            continue
        out.append(
            IntegrationSegment(
                start_chainage_m=float(s[idx[0]]),
                end_chainage_m=float(s[idx[-1]]),
                section_count=len(idx),
                volume_m3=trapezoid(a[idx], s[idx]),
                mean_area_m2=float(a[idx].mean()),
            )
        )
    return out


def summarise_coverage(
    series, segments: list[IntegrationSegment], ranges: list[Interval] | None = None
) -> CoverageReport:
    """Coverage of *segments* against what was requested (the holdout ranges, or the series)."""
    s, a = _stations(series)
    requested = _requested(s, ranges)
    # Clipped to what was requested, so covered and missing partition it exactly. A station may
    # sit an epsilon outside a range and still be picked up; that must not make coverage > 1.
    covered = subtract_intervals(
        requested,
        subtract_intervals(
            requested, merge_intervals([(g.start_chainage_m, g.end_chainage_m) for g in segments])
        ),
    )
    missing = subtract_intervals(requested, covered)
    req_len, cov_len = total_length(requested), total_length(covered)
    inside = np.zeros(len(s), bool)
    for lo, hi in requested:
        inside |= (s >= lo - EPS_M) & (s <= hi + EPS_M)
    observed = inside & np.isfinite(a)
    return CoverageReport(
        requested_intervals_m=requested,
        integrated_intervals_m=covered,
        missing_intervals_m=missing,
        requested_length_m=req_len,
        covered_length_m=cov_len,
        # Nothing requested is not full coverage of nothing: a zero-length span means the
        # caller asked for a volume over no tunnel, and reporting 1.0 would read as complete.
        coverage_fraction=(cov_len / req_len) if req_len > EPS_M else 0.0,
        valid_section_count=int(observed.sum()),
        missing_section_count=int((inside & ~observed).sum()),
        missing_chainages_m=[float(v) for v in s[inside & ~observed]],
    )
