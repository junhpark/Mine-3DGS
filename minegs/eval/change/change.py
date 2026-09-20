"""Epoch differencing (§11 change): ΔA(s) over the common chainage range -> ΔV.

``diff_sections`` is arithmetic over two section series: it does **not** establish that the
two epochs are comparable (same frame, same scale basis, same reference axis, leak-free
ranges, different epoch ids). That is the epoch-pair protocol ``judge_change``, Phase 7
(docs/ROADMAP.md). Until it exists the result is labelled ``geometry_diagnostic`` and must
not be reported as a validated change volume.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.errors import ContractError
from minegs.eval.sections.sections import SectionSeries
from minegs.eval.volume.coverage import Interval, segmented_integral, subtract_intervals


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
    #: The chainage spans ΔV was actually integrated over, and the spans inside the common range
    #: that neither epoch covered. ΔV is the sum over the first list and says nothing about the
    #: second (Phase 1C).
    differenced_intervals_m: list[Interval] = Field(default_factory=list)
    missing_intervals_m: list[Interval] = Field(default_factory=list)
    # never "change_volume": that claim needs the Phase 7 epoch-pair protocol (§5)
    claim: str = "geometry_diagnostic"


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
    # Gap-safe, for the same reason ``integrate_sections`` is (Phase 1C): a station missing from
    # either epoch means nobody measured that span twice, and a trapezoid across it would report
    # a difference over tunnel neither epoch observed.
    # ``max_step`` is what makes a station that only one epoch has count as a gap: the
    # intersection above drops it, leaving its neighbours adjacent and the trapezoid running
    # straight across a span one of the two epochs never measured.
    dv, spans = segmented_integral(common.astype(float), d, max_step=a.interval_m)
    return ChangeReport(
        epoch_a=epoch_a,
        epoch_b=epoch_b,
        start_chainage_m=float(common[0]),
        end_chainage_m=float(common[-1]),
        section_interval_m=a.interval_m,
        chainage_m=[float(s) for s in common],
        delta_area_m2=delta,
        delta_volume_m3=dv,
        valid_section_count=int(m.sum()),
        missing_section_count=int((~m).sum()),
        reference_axis=reference_axis,
        differenced_intervals_m=spans,
        missing_intervals_m=subtract_intervals(
            [(float(common[0]), float(common[-1]))] if len(common) >= 2 else [], spans
        ),
    )
