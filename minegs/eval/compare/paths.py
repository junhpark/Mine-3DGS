"""TLS-assisted against image-only, on one domain (§Phase 3 C3, §14).

The question Phase 3 exists to answer is not "how good is the image-only reconstruction" but
"how does it compare with the TLS-assisted one" — and that comparison is worthless unless both
sides are asked the same question. Two reconstructions cover different pieces of a drift; a
volume error computed over 38 m and one computed over 22 m are not two measurements of the same
thing, and the shorter one flatters itself.

So this refuses first and compares second:

* both predictions must be cut on the same grid — same reference axis, same stations, same
  interval, slab and bin count — or the areas are areas of different things;
* both are then integrated over the *intersection* of what they each observed, which is
  smaller than either and is the only domain on which the two numbers mean the same thing;
* what each side left out is reported, not dropped.

Nothing here integrates. Both numbers come from ``compare_to_reference``, the Phase 1C/2 helper,
called a second time with the common intervals as its requested range — a second implementation
of the volume would be a second answer waiting to disagree with the first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from minegs.core.config import VersionedModel
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest
from minegs.core.provenance import ProvenanceRecord, sha256_tree, stamp
from minegs.eval.volume.paired import (
    Interval,
    PairedValidation,
    compare_to_reference,
    require_same_grid,
)

COMPARISON_FILE = "path_comparison.json"

#: What each path has to have declared before its numbers can be called real, by label. Every
#: key must be present *and* true: a stage nobody reported is "we do not know", and we do not
#: know is not "it ran". The two lists differ because the scanner path has no SfM and no frame
#: decoder, so requiring those of it would make a real TLS survey unprovable.
REQUIRED_EXECUTION: dict[str, tuple[str, ...]] = {
    "tls_assisted": ("real_gpu_execution", "real_renderer_execution"),
    "image_only": (
        "real_frame_extraction",
        "real_sfm_execution",
        "real_gpu_execution",
        "real_renderer_execution",
    ),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathResult(_Strict):
    """One reconstruction path's numbers, over the common domain, with how it was produced."""

    label: str
    dataset_id: str
    dataset_hash: str
    source: str
    initialization_source: str
    paired: PairedValidation
    #: One flag per stage that can be substituted, ``True`` only when that stage really ran.
    #: An empty dict is not "nothing was substituted" — it is "nobody said", and it counts
    #: against ``real_execution`` exactly as a ``False`` does.
    execution: dict[str, bool] = Field(default_factory=dict)
    registration: dict[str, Any] | None = None
    claims: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PathComparison(VersionedModel):
    """``path_comparison.json`` — both paths, one domain, and what is missing from it."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    comparison_id: str = Field(min_length=1)
    grid: dict[str, Any] = Field(default_factory=dict)
    requested_intervals_m: list[Interval] = Field(default_factory=list)
    #: Where both sides observed. Every number below is over exactly this.
    common_intervals_m: list[Interval] = Field(default_factory=list)
    common_length_m: float = 0.0
    #: Requested minus common, and which side was missing there.
    excluded_intervals_m: list[Interval] = Field(default_factory=list)
    tls_assisted: PathResult
    image_only: PathResult
    #: The difference the comparison exists for, over the common domain only.
    volume_difference_m3: float | None = None
    section_median_difference_m2: float | None = None
    #: True only when **each** path separately declared every stage it needs and declared
    #: them all real. Anything substituted, or undeclared, leaves this False and these numbers
    #: structural rather than scientific.
    real_execution: bool = False
    maturity_statement: str = ""
    notes: list[str] = Field(default_factory=list)
    provenance: ProvenanceRecord


MATURITY_PENDING = (
    "Phase 3 image/360 independent reconstruction path is implemented and structurally "
    "tested. Real-data scientific validation remains NOT VALIDATED. Phase 3 G2 remains "
    "PENDING."
)


def path_context(
    dataset_dir: str | Path, *, e2e_report: str | Path | None = None
) -> dict[str, Any]:
    """What one path's numbers rest on, read out of the dataset rather than asserted.

    Every execution flag is positive — ``True`` means that stage really ran. A stage nobody
    reported does not appear, and an absent flag is not a pass: ``compare_paths`` needs the
    flags it has to be unanimous *and* present before it calls a comparison real.
    """
    ds = Path(dataset_dir)
    manifest = Manifest.load_dataset(ds, strict_layout=False)
    from minegs.eval.protocol import judge
    from minegs.train.runner.base import DATASET_HASH_PATTERNS

    execution: dict[str, bool] = {}
    prov_file = ds / "provenance" / "phase3" / "init_provenance.json"
    if prov_file.is_file():
        prov = json.loads(prov_file.read_text())
        execution["real_sfm_execution"] = bool(prov.get("real_sfm_execution"))
        execution["real_frame_extraction"] = bool(prov.get("frame_extraction_real"))
    if e2e_report is not None:
        rep = json.loads(Path(e2e_report).read_text())
        execution["real_gpu_execution"] = bool(rep.get("training", {}).get("real_gpu_execution"))
        execution["real_renderer_execution"] = bool(
            rep.get("reconstruction", {}).get("real_renderer_execution")
        )
    j = judge(manifest)
    ctx: dict[str, Any] = {
        "dataset_id": manifest.dataset_id,
        "dataset_hash": sha256_tree(ds, DATASET_HASH_PATTERNS),
        "source": manifest.source,
        "initialization_source": manifest.initialization.source,
        "execution": execution,
        "claims": [c.value for c in j.claims],
        "holdout_ranges_m": [tuple(r) for r in j.holdout_ranges_m],
    }
    reg = manifest.registration
    if reg is not None:
        ctx["registration"] = {
            "registration_id": reg.registration_id,
            "basis": reg.basis,
            "scale": reg.scale,
            "rmse_m": reg.rmse_m,
            "inlier_ratio": reg.inlier_ratio,
            "support_ranges_m": reg.support_ranges_m,
            "claim_allowed": reg.claim_allowed,
            "claim_refusals": list(reg.claim_refusals),
        }
    return ctx


def require_same_holdout(
    tls_context: dict[str, Any], image_context: dict[str, Any]
) -> list[Interval]:
    """The declared evaluation holdout, which both paths must have declared identically.

    Two reconstructions evaluated against different holdouts are two experiments, and the
    comparison would be between them rather than between the paths. Refusing here is cheaper
    than a plausible number nobody can interpret.
    """
    a = [tuple(r) for r in tls_context.get("holdout_ranges_m", [])]
    b = [tuple(r) for r in image_context.get("holdout_ranges_m", [])]
    if not a or not b:
        raise ContractError(
            "a path comparison is over a declared geometry holdout, and "
            f"{'the TLS-assisted' if not a else 'the image-only'} dataset declares none. Pass "
            "explicit ranges only if you mean to compare over something other than a holdout, "
            "and know the result is diagnostic."
        )
    if sorted(a) != sorted(b):
        raise ContractError(
            f"the two datasets declare different geometry holdouts ({sorted(a)} vs {sorted(b)}); "
            "comparing them would compare two experiments, not two reconstruction paths"
        )
    return sorted(a)


def _path_execution(label: str, flags: dict[str, bool]) -> tuple[bool, list[str], list[str]]:
    """Whether one path really ran, what it never reported, and what it substituted."""
    required = REQUIRED_EXECUTION[label]
    missing = [k for k in required if k not in flags]
    substituted = sorted(k for k, v in flags.items() if not v)
    return (not missing and not substituted), missing, substituted


def _intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for lo1, hi1 in a:
        for lo2, hi2 in b:
            lo, hi = max(lo1, lo2), min(hi1, hi2)
            if hi > lo:
                out.append((lo, hi))
    return sorted(out)


def _subtract(whole: list[Interval], part: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for lo, hi in whole:
        cuts = sorted((max(lo, a), min(hi, b)) for a, b in part if min(hi, b) > max(lo, a))
        cursor = lo
        for a, b in cuts:
            if a > cursor:
                out.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < hi:
            out.append((cursor, hi))
    return out


def _same_grid(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """The two paths must have asked the same geometric question.

    ``dataset_id`` is deliberately not compared: the whole point is two datasets. Everything
    that decides what "the area at 22 m" means is.
    """
    for key in (
        "reference_axis",
        "interval_m",
        "thickness_m",
        "angle_bins",
        "frame",
        "station_count",
    ):
        if a.get(key) != b.get(key):
            raise ContractError(
                f"the TLS-assisted and image-only paths disagree about {key} "
                f"({a.get(key)!r} vs {b.get(key)!r}). They were not cut on the same grid, so "
                "comparing their areas would compare areas of different things."
            )
    return dict(a)


def compare_paths(
    *,
    tls_pred: Any,
    tls_ref: Any,
    image_pred: Any,
    image_ref: Any,
    ranges: list[Interval],
    tls_context: dict[str, Any],
    image_context: dict[str, Any],
    comparison_id: str,
) -> PathComparison:
    """Compare two reconstruction paths over the domain both of them observed.

    ``*_pred`` and ``*_ref`` are ``SectionRecord``s: each path's prediction and the TLS
    reference it is measured against. ``*_context`` carries what the numbers rest on — the
    dataset's identity, its claims, which of its stages really ran and, for the image-only
    path, its registration.
    """
    grid_tls = require_same_grid(tls_pred, tls_ref)
    grid_img = require_same_grid(image_pred, image_ref)
    grid = _same_grid(grid_tls, grid_img)

    first_tls = compare_to_reference(tls_pred, tls_ref, ranges)
    first_img = compare_to_reference(image_pred, image_ref, ranges)
    common = _intersect(
        first_tls.volume.integrated_intervals_m, first_img.volume.integrated_intervals_m
    )
    excluded = _subtract([tuple(r) for r in ranges], common)

    # No fallback to each path's own domain when the intersection is empty: falling back is
    # exactly the subtraction this module exists to refuse — two volumes over two different
    # pieces of tunnel, differenced. An empty domain gives empty reports and no difference.
    paired_tls = compare_to_reference(tls_pred, tls_ref, common)
    paired_img = compare_to_reference(image_pred, image_ref, common)

    notes: list[str] = []
    if not common:
        notes.append(
            "the two paths share no observed chainage; there is no domain on which their "
            "numbers are about the same tunnel, so no difference is reported"
        )
    if excluded:
        notes.append(
            f"{len(excluded)} interval(s) were requested and are not in the comparison, because "
            "at least one path did not observe them"
        )

    exec_tls = dict(tls_context.get("execution", {}))
    exec_img = dict(image_context.get("execution", {}))
    # Judged per path, never on a merged dictionary. Merging let one path's flag overwrite the
    # other's under the same key — an image path that really trained could erase the fact that
    # the TLS path's trainer was substituted — and it let a single true flag stand in for a
    # list nobody checked was complete.
    real_tls, missing_tls, subs_tls = _path_execution("tls_assisted", exec_tls)
    real_img, missing_img, subs_img = _path_execution("image_only", exec_img)
    real = real_tls and real_img
    if not real:
        for label, missing, substituted in (
            ("the TLS-assisted path", missing_tls, subs_tls),
            ("the image-only path", missing_img, subs_img),
        ):
            if substituted:
                notes.append(f"{label} substituted {', '.join(substituted)}")
            if missing:
                notes.append(f"{label} did not say whether {', '.join(missing)} really ran")
        notes.append(
            "so this comparison is structural evidence about the pipeline and not a "
            "measurement of a mine"
        )

    vol_tls = paired_tls.volume.predicted_volume_m3
    vol_img = paired_img.volume.predicted_volume_m3
    diff = None if (vol_tls is None or vol_img is None) else float(vol_img - vol_tls)
    med_tls = paired_tls.sections.median_absolute_error_m2
    med_img = paired_img.sections.median_absolute_error_m2
    med_diff = None if (med_tls is None or med_img is None) else float(med_img - med_tls)

    return PathComparison(
        comparison_id=comparison_id,
        grid=grid,
        requested_intervals_m=[tuple(r) for r in ranges],
        common_intervals_m=common,
        common_length_m=float(sum(hi - lo for lo, hi in common)),
        excluded_intervals_m=excluded,
        tls_assisted=PathResult(
            label="tls_assisted",
            dataset_id=tls_context["dataset_id"],
            dataset_hash=tls_context["dataset_hash"],
            source=tls_context.get("source", "tls"),
            initialization_source=tls_context.get("initialization_source", "tls"),
            paired=paired_tls,
            execution=exec_tls,
            claims=list(tls_context.get("claims", [])),
        ),
        image_only=PathResult(
            label="image_only",
            dataset_id=image_context["dataset_id"],
            dataset_hash=image_context["dataset_hash"],
            source=image_context.get("source", "video"),
            initialization_source=image_context.get("initialization_source", "sfm_sparse"),
            paired=paired_img,
            execution=exec_img,
            registration=image_context.get("registration"),
            claims=list(image_context.get("claims", [])),
        ),
        volume_difference_m3=diff,
        section_median_difference_m2=med_diff,
        real_execution=real,
        maturity_statement=MATURITY_PENDING,
        notes=notes,
        provenance=stamp({"ranges": [list(r) for r in ranges]}),
    )


__all__ = [
    "COMPARISON_FILE",
    "MATURITY_PENDING",
    "REQUIRED_EXECUTION",
    "PathComparison",
    "PathResult",
    "compare_paths",
    "path_context",
    "require_same_holdout",
]
