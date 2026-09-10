"""Epoch differencing (§11 change): ΔA(s) over the common chainage range -> ΔV."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from minegs.core.errors import ContractError
from minegs.eval.sections.sections import SectionSeries


class ChangeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epoch_a: str
    epoch_b: str
    start_chainage_m: float
    end_chainage_m: float
    section_interval_m: float
    chainage_m: list[float]
    delta_area_m2: list[float | None]
    delta_volume_m3: float  # positive = b larger than a (excavation)
    valid_section_count: int
    missing_section_count: int
    reference_axis: str


def diff_sections(
    a: SectionSeries, b: SectionSeries, epoch_a: str, epoch_b: str, reference_axis: str
) -> ChangeReport:
    if abs(a.interval_m - b.interval_m) > 1e-9:
        raise ContractError("section intervals differ between epochs")
    sa, sb = a.chainages(), b.chainages()
    common = np.intersect1d(np.round(sa, 6), np.round(sb, 6))
    if len(common) < 2:
        raise ContractError("epochs share fewer than 2 section chainages")
    ia = {round(float(s), 6): i for i, s in enumerate(sa)}
    ib = {round(float(s), 6): i for i, s in enumerate(sb)}
    aa, ab = a.areas(), b.areas()
    delta: list[float | None] = []
    for s in common:
        va, vb = aa[ia[float(s)]], ab[ib[float(s)]]
        delta.append(None if np.isnan(va) or np.isnan(vb) else float(vb - va))
    d = np.array([v if v is not None else np.nan for v in delta])
    m = ~np.isnan(d)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return ChangeReport(
        epoch_a=epoch_a,
        epoch_b=epoch_b,
        start_chainage_m=float(common[0]),
        end_chainage_m=float(common[-1]),
        section_interval_m=a.interval_m,
        chainage_m=[float(s) for s in common],
        delta_area_m2=delta,
        delta_volume_m3=float(trap(d[m], common[m])) if m.sum() >= 2 else 0.0,
        valid_section_count=int(m.sum()),
        missing_section_count=int((~m).sum()),
        reference_axis=reference_axis,
    )
