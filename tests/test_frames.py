import numpy as np
import pytest
from minegs.core.errors import FrameError
from minegs.core.frames import (
    SE3,
    Sim3,
    check_float32_safe,
    float32_precision_m,
    quat_to_rotmat,
    rot_x,
    rot_y,
    rot_z,
    rotmat_to_quat,
)


def test_se3_roundtrip_and_compose(rng):
    R = rot_z(30) @ rot_y(-12) @ rot_x(5)
    T = SE3(R, [1.0, -2.0, 3.0])
    p = rng.normal(size=(50, 3))
    assert np.allclose(T.inverse().apply(T.apply(p)), p)
    assert (T @ T.inverse()).is_identity()
    assert SE3.from_matrix(T.matrix()).allclose(T)
    assert np.allclose(quat_to_rotmat(rotmat_to_quat(R)), R)


def test_se3_rejects_non_rigid():
    with pytest.raises(FrameError):
        SE3(np.eye(3) * 2, np.zeros(3))
    M = np.eye(4)
    M[3, 0] = 1
    with pytest.raises(FrameError):
        SE3.from_matrix(M)


def test_sim3_scale_and_inverse(rng):
    T = Sim3(1.0032, rot_z(7), [0.5, 0.1, -0.2])
    p = rng.normal(size=(20, 3))
    assert np.allclose(T.inverse().apply(T.apply(p)), p)
    assert np.isclose(Sim3.from_matrix(T.matrix()).s, 1.0032)
    with pytest.raises(FrameError):
        T.se3()  # scale must be resolved explicitly
    assert T.se3_part().R.shape == (3, 3)
    assert (T @ SE3.identity()).s == pytest.approx(1.0032)


def test_float32_precision_argument():
    # §3: UTM coordinates (~4e6 m) in float32 lose ~0.5 m precision
    assert float32_precision_m(4_012_300.0) > 0.25
    with pytest.raises(FrameError):
        check_float32_safe(np.array([[318300.0, 4012300.0, 120.0]]))
    assert check_float32_safe(np.array([[30.0, -40.0, 1.0]])) == 40.0
