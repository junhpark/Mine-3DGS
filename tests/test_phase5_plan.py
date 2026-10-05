"""Phase 5 C1 — the chunk plan: ownership, support, atomic capture groups, identity.

Negative tests forge a written plan on purpose and expect the verifier to name what is wrong.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from minegs.chunks.plan import (
    PLAN_FILE,
    build_chunk_plan,
    check_policy,
    load_chunk_plan,
    plan_content,
    verify_chunk_plan,
)
from minegs.core.errors import ContractError
from minegs.core.manifest import Manifest
from minegs.core.provenance import sha256_tree
from minegs.train.runner.base import DATASET_HASH_PATTERNS

from phase5_scene import CORE_M, HOLDOUT, OVERLAP_M, long_tunnel


@pytest.fixture(scope="module")
def tunnel(tmp_path_factory):
    sc = long_tunnel(tmp_path_factory.mktemp("p5"))
    sc.plan, sc.plan_path = build_chunk_plan(sc.dataset_dir, CORE_M, OVERLAP_M)
    sc.manifest = Manifest.load_dataset(sc.dataset_dir)
    return sc


@pytest.fixture(scope="module")
def excluded(tmp_path_factory):
    return long_tunnel(tmp_path_factory.mktemp("p5x"), images_excluded=True)


def _forge(tunnel, tmp_path, edit) -> Path:
    """A copy of the written plan, edited as plain JSON."""
    dst = tmp_path / "plan"
    dst.mkdir()
    data = json.loads(tunnel.plan_path.read_text())
    edit(data)
    (dst / PLAN_FILE).write_text(json.dumps(data))
    return dst


# ================================================================ the plan itself


def test_three_chunks_with_core_ownership_and_overlapping_support(tunnel):
    p = tunnel.plan
    assert [c.chunk_id for c in p.chunks] == ["K000", "K001", "K002"]
    s0, s1 = p.axis_range_m
    assert p.chunks[0].core_range_m == (s0, 100.0) and p.chunks[0].support_range_m == (s0, 120.0)
    assert p.chunks[1].core_range_m == (100.0, 200.0)
    assert p.chunks[1].support_range_m == (80.0, 220.0)
    assert p.chunks[2].core_range_m == (200.0, s1) and p.chunks[2].support_range_m == (180.0, s1)
    # cores tile the axis; supports overlap, cores do not
    for a, b in zip(p.chunks, p.chunks[1:], strict=False):
        assert a.core_range_m[1] == b.core_range_m[0]
        assert a.support_range_m[1] > b.support_range_m[0]


def test_every_chainage_has_exactly_one_owner(tunnel):
    p = tunnel.plan
    s = np.arange(p.axis_range_m[0], p.axis_range_m[1] + 1e-9, 0.5)
    owner = p.owner_index(s)
    for i, c in enumerate(p.chunks):
        lo, hi = c.core_range_m
        last = i == len(p.chunks) - 1
        mine = s[owner == i]
        assert np.all(mine >= lo) and (np.all(mine <= hi) if last else np.all(mine < hi))
    # a boundary belongs to the chunk after it, the axis end to the last chunk
    assert list(p.owner_index([99.999, 100.0, 200.0, p.axis_range_m[1]])) == [0, 1, 2, 2]


def test_same_input_same_plan_and_writing_it_keeps_the_dataset_identity(tunnel, tmp_path):
    before = sha256_tree(tunnel.dataset_dir, DATASET_HASH_PATTERNS)
    again, path = build_chunk_plan(tunnel.dataset_dir, CORE_M, OVERLAP_M)
    assert again.plan_digest == tunnel.plan.plan_digest and path == tunnel.plan_path
    assert sha256_tree(tunnel.dataset_dir, DATASET_HASH_PATTERNS) == before
    assert tunnel.plan.dataset_hash == before
    a = plan_content(tunnel.dataset_dir, CORE_M, OVERLAP_M)
    b = plan_content(tunnel.dataset_dir, CORE_M, OVERLAP_M)
    assert a == b
    other, _ = build_chunk_plan(tunnel.dataset_dir, 150.0, 10.0, tmp_path / "other")
    assert other.plan_id != tunnel.plan.plan_id


@pytest.mark.parametrize(
    ("core", "overlap", "match"),
    [
        (0.0, 0.0, "core_length_m must be a positive"),
        (-10.0, 0.0, "core_length_m must be a positive"),
        (float("nan"), 0.0, "core_length_m must be a positive"),
        (100.0, -1.0, "overlap_m must be >= 0"),
        (100.0, 100.0, "must be smaller than core_length_m"),
        (100.0, 150.0, "must be smaller than core_length_m"),
    ],
)
def test_invalid_core_length_or_overlap_is_refused(tunnel, core, overlap, match):
    with pytest.raises(ContractError, match=match):
        check_policy(core, overlap)
    with pytest.raises(ContractError, match=match):
        build_chunk_plan(tunnel.dataset_dir, core, overlap)


def test_a_remainder_shorter_than_the_overlap_joins_the_last_core(tunnel):
    # The axis is a little longer than 300 m: no 0.06 m chunk of its own.
    assert tunnel.plan.axis_range_m[1] > 300.0 and len(tunnel.plan.chunks) == 3
    with pytest.raises(ContractError, match="no training image"):
        # with no overlap there is nothing to merge into, and the sliver has no image
        build_chunk_plan(tunnel.dataset_dir, CORE_M, 0.0)


def test_a_chunk_with_no_training_image_is_refused(tunnel):
    with pytest.raises(ContractError, match=r"K001 .* has no training image"):
        build_chunk_plan(tunnel.dataset_dir, 10.0, 0.0)


# ================================================================ forged plans


def _gap(d):
    d["chunks"][1]["core_range_m"] = [110.0, 200.0]


def _double(d):
    d["chunks"][0]["core_range_m"] = [0.0, 110.0]


def _support(d):
    d["chunks"][1]["support_range_m"] = [70.0, 230.0]


def _not_containing(d):
    d["chunks"][1]["support_range_m"] = [105.0, 220.0]


def _ids(d):
    d["chunks"][2]["chunk_id"] = "K001"


def _order(d):
    d["chunks"][0], d["chunks"][1] = d["chunks"][1], d["chunks"][0]


def _outside(d):
    d["chunks"][2]["core_range_m"] = [200.0, 400.0]


def _content(d):
    d["chunks"][0]["images"] = d["chunks"][0]["images"][1:]


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (_gap, "gap between K000"),
        (_double, "owned twice"),
        (_support, "not its core widened by the overlap"),
        (_not_containing, "does not contain its core"),
        (_ids, "chunk ids repeat"),
        (_order, "ordinals are not 0..n-1"),
        (_outside, "core outside the axis"),
        (_content, "does not hash to its plan_digest"),
    ],
)
def test_a_forged_plan_is_refused(tunnel, tmp_path, edit, match):
    with pytest.raises(ContractError, match=match):
        verify_chunk_plan(tunnel.dataset_dir, _forge(tunnel, tmp_path, edit))


def test_a_resealed_plan_that_is_not_this_dataset_s_plan_is_refused(tunnel, tmp_path):
    """Digest made consistent again: the re-derivation still finds the difference."""
    import hashlib

    from minegs.core.config import canonical_json

    def edit(d):
        d["chunks"][0]["images"] = d["chunks"][0]["images"][1:]
        content = {k: v for k, v in d.items() if k not in ("plan_id", "plan_digest", "provenance")}
        d["plan_digest"] = hashlib.sha256(canonical_json(content).encode()).hexdigest()
        d["plan_id"] = "cplan_" + d["plan_digest"][:12]

    with pytest.raises(ContractError, match="not the plan this dataset and policy produce"):
        verify_chunk_plan(tunnel.dataset_dir, _forge(tunnel, tmp_path, edit))


def test_a_changed_dataset_refuses_the_plan(tunnel, tmp_path):
    ds = tmp_path / "ds"
    shutil.copytree(tunnel.dataset_dir, ds)
    verify_chunk_plan(ds, tunnel.plan_path)
    img = sorted((ds / "images").iterdir())[0]
    img.write_bytes(img.read_bytes() + b"\0")
    with pytest.raises(ContractError, match="dataset hash"):
        verify_chunk_plan(ds, tunnel.plan_path)


def test_a_changed_centerline_refuses_the_plan(tunnel, tmp_path):
    ds = tmp_path / "ds"
    shutil.copytree(tunnel.dataset_dir, ds)
    m = Manifest.load_dataset(ds)
    cl = ds / m.centerline.file
    cl.write_text(cl.read_text() + "# moved\n")
    with pytest.raises(ContractError, match="chainage means something else"):
        verify_chunk_plan(ds, tunnel.plan_path)


def test_a_plan_of_another_dataset_is_refused(tunnel, excluded):
    with pytest.raises(ContractError, match=r"dataset hash|not the plan"):
        verify_chunk_plan(excluded.dataset_dir, tunnel.plan_path)


# ================================================================ capture groups


def test_capture_groups_are_atomic_and_cross_boundaries_whole(tunnel):
    m = tunnel.manifest
    train = set(m.train_images()) - set(m.test_images())
    for c in tunnel.plan.chunks:
        for g in c.capture_groups:
            members = [x for x in m.capture_groups[g].members if x in train]
            assert set(members) <= set(c.images), f"{g} split in {c.chunk_id}"
        assert set(c.images) == {
            x for g in c.capture_groups for x in m.capture_groups[g].members if x in train
        }
    # the 75-165 m video segment crosses the 100 m boundary: whole in both chunks
    k0, k1, k2 = tunnel.plan.chunks
    assert "V001" in k0.capture_groups and "V001" in k1.capture_groups
    assert "V001" not in k2.capture_groups
    # and the image support says so instead of hiding it
    assert k0.actual_image_support_m[1] == 165.0 > k0.support_range_m[1]


def test_the_global_split_comes_first(tunnel):
    m = tunnel.manifest
    test = set(m.test_images())
    for c in tunnel.plan.chunks:
        assert not test & set(c.images)
        assert not set(m.split.test_groups) & set(c.capture_groups)
        # test groups are rendered (surface), never trained
        assert set(c.view_groups) >= set(c.capture_groups)
    assert any(set(m.split.test_groups) & set(c.view_groups) for c in tunnel.plan.chunks)


def test_images_the_holdout_removed_from_training_do_not_come_back(excluded):
    m = Manifest.load_dataset(excluded.dataset_dir)
    gone = set(m.images_of(m.split.train_groups)) - set(m.train_images())
    assert gone, "the fixture should exclude the holdout station"
    plan, _ = build_chunk_plan(excluded.dataset_dir, CORE_M, OVERLAP_M)
    for c in plan.chunks:
        assert not gone & set(c.images)
    lo, hi = HOLDOUT
    assert all(
        not (m.capture_groups[g].span()[1] >= lo and m.capture_groups[g].span()[0] <= hi)
        for c in plan.chunks
        for g in c.capture_groups
        if m.capture_groups[g].type == "tls_station"
    )


def test_a_train_group_without_chainage_is_refused(tunnel, tmp_path):
    ds = tmp_path / "ds"
    shutil.copytree(tunnel.dataset_dir, ds)
    m = Manifest.load_dataset(ds)
    gid = m.split.train_groups[0]
    m.capture_groups[gid].chainage_m = None
    m.capture_groups[gid].chainage_range_m = None
    m.save_dataset(ds)
    with pytest.raises(ContractError, match=f"train group {gid} has no chainage span"):
        build_chunk_plan(ds, CORE_M, OVERLAP_M, tmp_path / "plan")


def test_group_selection_is_deterministic_and_covers_every_training_image(tunnel):
    m = tunnel.manifest
    train = set(m.train_images()) - set(m.test_images())
    assert set().union(*(set(c.images) for c in tunnel.plan.chunks)) == train
    for c in tunnel.plan.chunks:
        assert c.images == sorted(c.images) and c.capture_groups == sorted(c.capture_groups)


def test_video_groups_stay_whole(tmp_path):
    """An image-only dataset (plain video, consecutive-frame groups) cut on short cores."""
    from phase4_scene import video_scene

    sc = video_scene(tmp_path / "v")
    plan, _ = build_chunk_plan(sc.dataset_dir, 20.0, 5.0)
    m = Manifest.load_dataset(sc.dataset_dir)
    train = set(m.train_images()) - set(m.test_images())
    for c in plan.chunks:
        for g in c.capture_groups:
            assert {x for x in m.capture_groups[g].members if x in train} <= set(c.images)
    assert len(plan.chunks) >= 2


# ================================================================ CLI


def test_cli_writes_a_plan_and_refuses_one_in_the_manifest(tunnel, tmp_path):
    from minegs.cli.main import app
    from typer.testing import CliRunner

    r = CliRunner().invoke(
        app,
        [
            "dataset",
            "chunk-plan",
            str(tunnel.dataset_dir),
            "--core-length-m",
            "100",
            "--overlap-m",
            "20",
        ],
    )
    assert r.exit_code == 0, r.output
    assert tunnel.plan.plan_id in r.output
    r = CliRunner().invoke(
        app, ["dataset", "chunk-plan-verify", str(tunnel.dataset_dir), str(tunnel.plan_path)]
    )
    assert r.exit_code == 0 and "verified" in r.output
    r = CliRunner().invoke(app, ["dataset", "chunks", str(tunnel.dataset_dir), "--write"])
    assert r.exit_code != 0 and "dataset hash" in r.output
    assert load_chunk_plan(tunnel.plan_path).plan_id == tunnel.plan.plan_id
