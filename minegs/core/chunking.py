"""Chunking along centerline chainage (§10), not an XYZ grid.

v1 reserves chunks in the manifest; files are not physically split. Each chunk gets its own
LOCAL_METRIC origin (``T_tls_from_local``) at the centerline point mid-chunk, so per-chunk
training stays float32-safe on kilometre-long drifts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from minegs.core.errors import ContractError
from minegs.core.frames import SE3

if TYPE_CHECKING:
    from minegs.core.centerline import Centerline


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    range_m: tuple[float, float]
    T_tls_from_local: list[list[float]] | None = None
    groups: list[str] = Field(default_factory=list)

    @property
    def start(self) -> float:
        return self.range_m[0]

    @property
    def end(self) -> float:
        return self.range_m[1]

    def contains(self, s: float, margin: float = 0.0) -> bool:
        return (self.start - margin) <= s <= (self.end + margin)


class ChunkPlan(BaseModel):
    # manifest key is "list" (§4); the python attribute is ``items`` to avoid shadowing the type
    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)
    basis: str = "centerline_chainage"
    length_m: float
    overlap_m: float
    items: list[Chunk] = Field(default_factory=list, alias="list")


def plan_chunks(
    s_start: float,
    s_end: float,
    length_m: float,
    overlap_m: float,
    centerline: Centerline | None = None,
    prefix: str = "C",
) -> ChunkPlan:
    if length_m <= 0 or overlap_m < 0 or overlap_m >= length_m:
        raise ContractError("need 0 <= overlap_m < length_m > 0")
    if s_end <= s_start:
        raise ContractError("s_end must exceed s_start")
    step = length_m - overlap_m
    chunks: list[Chunk] = []
    a = s_start
    i = 1
    while True:
        b = min(a + length_m, s_end)
        c = Chunk(id=f"{prefix}{i:02d}", range_m=(round(a, 3), round(b, 3)))
        if centerline is not None:
            mid = centerline.point_at(0.5 * (a + b))
            c.T_tls_from_local = SE3.from_translation(mid).to_list()
        chunks.append(c)
        if b >= s_end - 1e-9:
            break
        a += step
        i += 1
    return ChunkPlan(length_m=length_m, overlap_m=overlap_m, items=chunks)


def assign_groups(plan: ChunkPlan, group_chainage: dict[str, tuple[float, float]]) -> ChunkPlan:
    """Attach capture groups whose chainage span intersects each chunk (incl. overlap)."""
    for c in plan.items:
        c.groups = sorted(
            g for g, (lo, hi) in group_chainage.items() if hi >= c.start and lo <= c.end
        )
    return plan


def chunk_membership(plan: ChunkPlan, s: np.ndarray) -> list[np.ndarray]:
    """Boolean mask per chunk over an array of chainages."""
    s = np.asarray(s, dtype=np.float64)
    return [(s >= c.start) & (s <= c.end) for c in plan.items]
