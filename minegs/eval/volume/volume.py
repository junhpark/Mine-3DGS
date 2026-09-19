"""Volume (§11): default ∫A(s)ds over the section series (mining practice for over/underbreak),
closed-mesh volume as secondary. ``volume.json`` must carry start/end chainage, section
interval, valid/missing counts and the reference axis.

Integration is *gap-safe* (Phase 1C): a run of consecutive observed stations is integrated,
a gap between two runs is reported as a missing interval and contributes nothing. See
``minegs/eval/volume/coverage.py`` for why, and for the interval arithmetic.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.errors import ContractError
from minegs.eval.sections.models import SectionSource
from minegs.eval.sections.sections import SectionSeries, polygon_area
from minegs.eval.volume.coverage import (
    CoverageReport,
    IntegrationSegment,
    Interval,
    plan_integration,
    trapezoid,
)


class VolumeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = "integrate_sections"
    #: The *envelope* of the integrated segments, not the integrated span: with two holdout
    #: ranges or a gap, ``end - start`` is longer than what was measured. The integrated length
    #: is ``coverage.covered_length_m``; the spans are ``coverage.integrated_intervals_m``.
    start_chainage_m: float
    end_chainage_m: float
    section_interval_m: float
    valid_section_count: int
    missing_section_count: int
    reference_axis: str  # e.g. "centerline:design:raw/centerline.csv"
    volume_m3: float
    #: Length-weighted over the integrated span, so ``mean_area_m2 * coverage.covered_length_m``
    #: is ``volume_m3`` exactly. The unweighted mean of the station areas does not reproduce the
    #: volume, and a reader who multiplies it by the chainage envelope gets a number for tunnel
    #: that was never integrated.
    mean_area_m2: float
    missing_chainages_m: list[float] = Field(default_factory=list)
    frame: str = "TLS_GLOBAL"
    claim: str = "geometry_diagnostic"
    mesh_volume_m3: float | None = None
    # ---- Phase 1C: what was integrated, and over what it was asked for.
    #: Which spans of tunnel the volume actually accounts for, and which it does not. A single
    #: number plus a count of missing sections never said whether 40 m³ covered the whole
    #: holdout or the first third of it.
    coverage: CoverageReport = Field(default_factory=CoverageReport)
    #: One entry per contiguous run of observed stations. ``volume_m3`` is their sum.
    segments: list[IntegrationSegment] = Field(default_factory=list)
    #: Provenance of the sections this was integrated from, when they came from an artifact
    #: rather than a bare series (``minegs/eval/sections/models.py``). Filled by the caller that
    #: holds the record; a volume whose source is unknown says so by leaving it null.
    section_id: str | None = None
    source: SectionSource | None = None
    #: How the sections were cut. ``section_interval_m`` alone does not say it: a 3 m slab and a
    #: 180-bin polygon are as much part of "what this number measured" as the interval, and a
    #: reader of volume.json should not have to open sections.json to see at what resolution the
    #: claim was made.
    section_parameters: dict | None = None


def _integrate(
    series: SectionSeries, ranges: list[Interval] | None
) -> tuple[list[IntegrationSegment], CoverageReport]:
    segments, coverage = plan_integration(series, ranges)
    if not segments:
        where = ", ".join(f"{lo:g}-{hi:g} m" for lo, hi in coverage.requested_intervals_m) or "the"
        raise ContractError(
            f"nothing to integrate over {where}: a volume needs at least two consecutive "
            f"observed sections, and this series has {coverage.valid_section_count} valid and "
            f"{coverage.missing_section_count} missing stations there. Sections are reported as "
            "missing, never interpolated (§11)."
        )
    return segments, coverage


def integrate_sections(
    series: SectionSeries,
    reference_axis: str,
    frame: str = "TLS_GLOBAL",
    ranges: list[Interval] | None = None,
) -> VolumeReport:
    """∫A(s)ds over the observed stations, segment by segment.

    *ranges* restricts the integration to declared spans — the geometry holdout, on a claim
    path. Passing ``None`` integrates the series' own span, which is the diagnostic case.
    """
    segments, coverage = _integrate(series, ranges)
    volume = float(sum(g.volume_m3 for g in segments))
    return VolumeReport(
        start_chainage_m=segments[0].start_chainage_m,
        end_chainage_m=segments[-1].end_chainage_m,
        section_interval_m=series.interval_m,
        valid_section_count=coverage.valid_section_count,
        missing_section_count=coverage.missing_section_count,
        reference_axis=reference_axis,
        volume_m3=volume,
        mean_area_m2=volume / coverage.covered_length_m,
        missing_chainages_m=list(coverage.missing_chainages_m),
        frame=frame,
        coverage=coverage,
        segments=segments,
    )


class DesignComparison(BaseModel):
    """Per-section Design vs actual: overbreak (actual outside design) / underbreak (inside)."""

    model_config = ConfigDict(extra="forbid")
    chainage_m: list[float]
    design_area_m2: float
    actual_area_m2: list[float | None]
    overbreak_m2: list[float | None]
    underbreak_m2: list[float | None]
    overbreak_m3: float
    underbreak_m3: float
    #: Same coverage contract as ``VolumeReport``: over/underbreak volumes are integrated over
    #: the observed runs only, never across a gap (Phase 1C).
    coverage: CoverageReport = Field(default_factory=CoverageReport)
    overbreak_segments_m3: list[float] = Field(default_factory=list)
    underbreak_segments_m3: list[float] = Field(default_factory=list)


def compare_to_design(
    series: SectionSeries,
    design_radii_m: np.ndarray | float,
    ranges: list[Interval] | None = None,
) -> DesignComparison:
    """``design_radii_m``: scalar (circular) or per-angle-bin radii matching ``series.angle_bins``.

    *ranges* restricts the comparison the same way it restricts ``integrate_sections``, and the
    per-section arrays are restricted with it: a table listing sections the volumes did not
    integrate would not describe the numbers underneath it.
    """
    n = series.angle_bins
    edges = np.linspace(0, 2 * np.pi, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    design = (
        np.full(n, float(design_radii_m))
        if np.isscalar(design_radii_m)
        else np.asarray(design_radii_m, dtype=np.float64)
    )
    if design.shape != (n,):
        raise ContractError(f"design radii must have {n} bins")
    d_area = polygon_area(design, centers)
    dtheta = 2 * np.pi / n

    # Over/underbreak per section is an area, so it is cut into contiguous runs exactly like
    # A(s). Reusing the segments from the *area* series is what guarantees the two reports agree
    # on which spans of tunnel were observed.
    segments, coverage = _integrate(series, ranges)
    inside = _stations_in(series, coverage.requested_intervals_m)
    ch, act, over, under = [], [], [], []
    for sec in [series.sections[i] for i in inside]:
        ch.append(sec.chainage_m)
        if not sec.valid or sec.area_m2 is None or not np.isfinite(sec.area_m2):
            act.append(None)
            over.append(None)
            under.append(None)
            continue
        r = np.array([v if v is not None else np.nan for v in sec.radii_m])
        good = ~np.isnan(r)
        if len(r) != n or not good.any():
            # np.interp with no sample points raises a bare ValueError from inside a claim
            # path. A section that reports an area while carrying no wall radii, or radii for a
            # different bin count, is a malformed series, and it should say so.
            raise ContractError(
                f"section at {sec.chainage_m:g} m has an area but {int(good.sum())} of "
                f"{len(r)} wall radii ({n} bins expected); over/underbreak is computed per "
                "angle bin and cannot be derived from that"
            )
        r = np.interp(centers, centers[good], r[good], period=2 * np.pi)
        # sector-wise area difference: 1/2 (r_a^2 - r_d^2) dθ
        diff = 0.5 * (r**2 - design**2) * dtheta
        act.append(sec.area_m2)
        over.append(float(np.clip(diff, 0, None).sum()))
        under.append(float(np.clip(-diff, 0, None).sum()))
    s = np.array(ch, dtype=np.float64)
    ov = np.array([v if v is not None else np.nan for v in over], dtype=np.float64)
    un = np.array([v if v is not None else np.nan for v in under], dtype=np.float64)
    ov_seg = [trapezoid(*_run(s, ov, g)) for g in segments]
    un_seg = [trapezoid(*_run(s, un, g)) for g in segments]
    return DesignComparison(
        chainage_m=ch,
        design_area_m2=d_area,
        actual_area_m2=act,
        overbreak_m2=over,
        underbreak_m2=under,
        overbreak_m3=float(sum(ov_seg)),
        underbreak_m3=float(sum(un_seg)),
        coverage=coverage,
        overbreak_segments_m3=ov_seg,
        underbreak_segments_m3=un_seg,
    )


def _stations_in(series: SectionSeries, intervals: list[Interval]) -> list[int]:
    """Indices of the sections inside *intervals*, ordered by chainage.

    By chainage rather than by list position: the arrays below are integrated against ``s``,
    and a series whose sections arrived out of order would otherwise trapezoid backwards.
    """
    from minegs.eval.volume.coverage import EPS_M

    s = series.chainages()
    keep = np.zeros(len(s), bool)
    for lo, hi in intervals:
        keep |= (s >= lo - EPS_M) & (s <= hi + EPS_M)
    idx = np.flatnonzero(keep)
    return [int(i) for i in idx[np.argsort(s[idx], kind="stable")]]


def _run(s: np.ndarray, y: np.ndarray, seg: IntegrationSegment) -> tuple[np.ndarray, np.ndarray]:
    """The (y, s) slice a segment covers, taken from the restricted per-section arrays."""
    from minegs.eval.volume.coverage import EPS_M

    m = (s >= seg.start_chainage_m - EPS_M) & (s <= seg.end_chainage_m + EPS_M)
    return y[m], s[m]
