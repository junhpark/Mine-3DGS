import numpy as np
import pytest
from minegs.core.centerline import Centerline
from minegs.core.chunking import assign_groups, plan_chunks
from minegs.core.errors import ContractError


def test_straight_centerline_basics():
    cl = Centerline([[0, 0, 0], [10, 0, 0], [10, 10, 0]])
    assert cl.length == pytest.approx(20)
    assert np.allclose(cl.point_at(15.0), [10, 5, 0])
    assert np.allclose(cl.tangent_at(5.0), [1, 0, 0])
    s, r = cl.project(np.array([[5.0, 2.0, 0.0], [12.0, 10.0, 1.0]]))
    assert s[0] == pytest.approx(5.0) and r[0] == pytest.approx(2.0)
    assert s[1] == pytest.approx(20.0) and r[1] == pytest.approx(np.hypot(2, 1))


def test_frame_at_is_right_handed_with_tangent_x():
    cl = Centerline([[0, 0, 0], [0, 10, 0]])
    T = cl.frame_at(3.0)
    assert np.allclose(T.R[:, 0], [0, 1, 0])
    assert np.allclose(np.cross(T.R[:, 0], T.R[:, 1]), T.R[:, 2])
    assert np.allclose(T.R[:, 2], [0, 0, 1])
    R, o = cl.frames_at(np.array([3.0, 4.0]))
    assert np.allclose(R[0], T.R) and np.allclose(o[0], T.t)


def test_csv_roundtrip_and_offset(tmp_path):
    cl = Centerline([[1, 2, 3], [4, 2, 3], [4, 8, 3]], chainage_offset_m=100.0)
    p = cl.to_csv(tmp_path / "c.csv")
    back = Centerline.from_csv(p)
    assert back.s_start == pytest.approx(100.0)
    assert np.allclose(back.vertices, cl.vertices)
    (tmp_path / "raw.csv").write_text("0,0,0\n5,0,0\n")
    assert Centerline.from_csv(tmp_path / "raw.csv").length == pytest.approx(5)


def test_extract_from_points_recovers_axis(rng):
    s = rng.uniform(0, 50, 20000)
    th = rng.uniform(0, 2 * np.pi, 20000)
    pts = np.column_stack([s, 2 * np.cos(th) + 0.02 * s, 2 * np.sin(th)])
    cl = Centerline.extract_from_points(pts, bin_m=2.0)
    assert cl.source == "extracted"
    assert 40 < cl.length < 52
    assert np.allclose(cl.vertices[:, 2], 0, atol=0.15)


def test_plan_chunks_with_overlap_and_groups():
    cl = Centerline([[0, 0, 0], [200, 0, 0]])
    plan = plan_chunks(0, 200, 80, 15, cl)
    assert [c.range_m for c in plan.items] == [(0, 80), (65, 145), (130, 200)]
    assert all(c.T_tls_from_local is not None for c in plan.items)
    plan = assign_groups(plan, {"S1": (10, 10), "S2": (70, 70), "V": (140, 190)})
    assert plan.items[0].groups == ["S1", "S2"]
    assert plan.items[1].groups == ["S2", "V"]
    assert plan.items[2].groups == ["V"]
    assert plan.model_dump(by_alias=True)["list"][0]["id"] == "C01"
    with pytest.raises(ContractError):
        plan_chunks(0, 10, 5, 5)
