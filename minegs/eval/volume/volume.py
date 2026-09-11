"""Volume (§11): default ∫A(s)ds over the section series (mining practice for over/underbreak),
closed-mesh volume as secondary. ``volume.json`` must carry start/end chainage, section
interval, valid/missing counts and the reference axis."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.errors import ContractError
from minegs.eval.sections.sections import SectionSeries, polygon_area


class VolumeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = "integrate_sections"
    start_chainage_m: float
    end_chainage_m: float
    section_interval_m: float
    valid_section_count: int
    missing_section_count: int
    reference_axis: str  # e.g. "centerline:design:raw/centerline.csv"
    volume_m3: float
    mean_area_m2: float
    missing_chainages_m: list[float] = Field(default_factory=list)
    frame: str = "TLS_GLOBAL"
    claim: str = "geometry_diagnostic"
    mesh_volume_m3: float | None = None


def integrate_sections(
    series: SectionSeries, reference_axis: str, frame: str = "TLS_GLOBAL"
) -> VolumeReport:
    s = series.chainages()
    a = series.areas()
    valid = ~np.isnan(a)
    if valid.sum() < 2:
        raise ContractError("need >= 2 valid sections to integrate")
    # trapezoid over valid sections only; gaps are integrated across (reported as missing)
    vol = (
        float(np.trapezoid(a[valid], s[valid]))
        if hasattr(np, "trapezoid")
        else float(np.trapz(a[valid], s[valid]))
    )
    return VolumeReport(
        start_chainage_m=float(s[valid][0]),
        end_chainage_m=float(s[valid][-1]),
        section_interval_m=series.interval_m,
        valid_section_count=int(valid.sum()),
        missing_section_count=int((~valid).sum()),
        reference_axis=reference_axis,
        volume_m3=vol,
        mean_area_m2=float(a[valid].mean()),
        missing_chainages_m=[float(v) for v in s[~valid]],
        frame=frame,
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


def compare_to_design(
    series: SectionSeries, design_radii_m: np.ndarray | float
) -> DesignComparison:
    """``design_radii_m``: scalar (circular) or per-angle-bin radii matching ``series.angle_bins``."""
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
    ch, act, over, under = [], [], [], []
    for sec in series.sections:
        ch.append(sec.chainage_m)
        if not sec.valid:
            act.append(None)
            over.append(None)
            under.append(None)
            continue
        r = np.array([v if v is not None else np.nan for v in sec.radii_m])
        good = ~np.isnan(r)
        r = np.interp(centers, centers[good], r[good], period=2 * np.pi)
        # sector-wise area difference: 1/2 (r_a^2 - r_d^2) dθ
        diff = 0.5 * (r**2 - design**2) * dtheta
        act.append(sec.area_m2)
        over.append(float(np.clip(diff, 0, None).sum()))
        under.append(float(np.clip(-diff, 0, None).sum()))
    s = np.array(ch)
    ov = np.array([v if v is not None else np.nan for v in over])
    un = np.array([v if v is not None else np.nan for v in under])
    m = ~np.isnan(ov)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return DesignComparison(
        chainage_m=ch,
        design_area_m2=d_area,
        actual_area_m2=act,
        overbreak_m2=over,
        underbreak_m2=under,
        overbreak_m3=float(trap(ov[m], s[m])) if m.sum() >= 2 else 0.0,
        underbreak_m3=float(trap(un[m], s[m])) if m.sum() >= 2 else 0.0,
    )
