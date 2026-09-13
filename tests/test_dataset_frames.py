"""Phase 0C §31 — frame contract: SOURCE → TLS_GLOBAL → LOCAL_METRIC."""

from __future__ import annotations

import numpy as np
import pytest
from minegs.core.errors import ContractError
from minegs.core.frames import SE3, rot_z
from minegs.dataset.build_config import (
    DatasetBuildConfig,
    LocalMetricConfig,
    SourceFrameConfig,
)
from minegs.dataset.frames import (
    build_frame_chain,
    check_frame_chain,
    local_origin,
    resolve_tls_from_source,
    validate_rigid,
)

BIG = np.array(
    [[420150.0, 3961420.0, 85.0], [420180.0, 3961440.0, 86.0], [420210.0, 3961470.0, 87.5]]
)


def _cfg(**over):
    d = {
        "dataset_id": "t",
        "source_frame": {"mode": "explicit_identity"},
        "camera": {"mode": "e57_pinhole", "R_e57cam_from_cam": np.eye(3).tolist()},
    }
    d.update(over)
    return d


# 1. SOURCE is never TLS_GLOBAL by omission
def test_source_frame_must_be_declared():
    d = _cfg()
    del d["source_frame"]
    with pytest.raises(Exception, match="source_frame"):
        DatasetBuildConfig.model_validate(d)
    with pytest.raises(Exception, match="explicit_transform"):
        SourceFrameConfig(mode="explicit_identity", T_tls_from_source=np.eye(4).tolist())
    with pytest.raises(Exception, match="needs T_tls_from_source"):
        SourceFrameConfig(mode="explicit_transform")


# 2. an explicit identity declaration is allowed and recorded as such
def test_explicit_identity_declaration():
    T = resolve_tls_from_source(SourceFrameConfig(mode="explicit_identity", note="survey frame"))
    assert T.is_identity()
    chain = build_frame_chain(SourceFrameConfig(mode="explicit_identity"), LocalMetricConfig(), BIG)
    assert (
        chain.source_mode == "explicit_identity"
        and chain.to_record()["T_tls_from_source"] == np.eye(4).tolist()
    )


# 3. a non-identity transform is applied through the chain
def test_non_identity_transform_is_applied():
    T = SE3(rot_z(30.0), [10.0, -5.0, 2.0])
    cfg = SourceFrameConfig(mode="explicit_transform", T_tls_from_source=T.to_list())
    assert resolve_tls_from_source(cfg).allclose(T)
    positions_tls = T.apply(BIG)
    chain = build_frame_chain(cfg, LocalMetricConfig(), positions_tls)
    p_local = chain.T_local_from_source.apply(BIG)
    expect = chain.T_local_from_tls.apply(T.apply(BIG))
    assert np.allclose(p_local, expect, atol=1e-9)


# 4. invalid transforms are refused
@pytest.mark.parametrize(
    "M, msg",
    [
        (np.eye(3), "4x4"),
        (np.eye(4) * np.nan, "non-finite"),
        (
            np.block([[np.eye(3) * 2, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]),
            "orthonormal",
        ),
        (np.vstack([np.eye(4)[:3], [0, 0, 0, 2]]), "last row"),
    ],
)
def test_invalid_transform_rejected(M, msg):
    with pytest.raises(ContractError, match=msg):
        validate_rigid(M, "T_tls_from_source")


# 5. a reflection is not a rotation
def test_reflection_rejected():
    M = np.eye(4)
    M[1, 1] = -1.0
    with pytest.raises(ContractError, match="reflection"):
        validate_rigid(M, "T_tls_from_source")
    with pytest.raises(Exception, match="rigid"):
        SourceFrameConfig(mode="explicit_transform", T_tls_from_source=M.tolist())


# 6. TLS -> LOCAL -> TLS is exact
def test_roundtrip_tls_local_tls():
    chain = build_frame_chain(SourceFrameConfig(mode="explicit_identity"), LocalMetricConfig(), BIG)
    back = chain.T_tls_from_local.apply(chain.T_local_from_tls.apply(BIG))
    assert np.allclose(back, BIG, atol=1e-9)


# 7. LOCAL_METRIC keeps scale 1 and no rotation
def test_local_metric_is_translation_only():
    chain = build_frame_chain(SourceFrameConfig(mode="explicit_identity"), LocalMetricConfig(), BIG)
    assert np.allclose(chain.T_tls_from_local.R, np.eye(3))
    assert abs(np.linalg.det(chain.T_tls_from_local.R) - 1.0) < 1e-12
    rotated = chain.__class__(
        chain.T_tls_from_source, SE3(rot_z(5.0), chain.T_tls_from_local.t), "x", "y"
    )
    with pytest.raises(ContractError, match="translation-only"):
        check_frame_chain(rotated, BIG)


# 8. the automatic origin is deterministic and order-independent
def test_automatic_origin_is_deterministic():
    cfg = LocalMetricConfig()
    a = local_origin(cfg, BIG)
    b = local_origin(cfg, BIG[::-1])
    assert np.array_equal(a, b)
    assert np.allclose(a, np.round(BIG.mean(axis=0) / 0.1) * 0.1)
    assert np.allclose(a * 10, np.round(a * 10))  # on the 0.1 m grid


# 9. an explicit origin is used verbatim
def test_explicit_origin():
    cfg = LocalMetricConfig(origin_policy="explicit", origin_tls=[420180.0, 3961440.0, 86.0])
    assert np.array_equal(local_origin(cfg, BIG), [420180.0, 3961440.0, 86.0])
    with pytest.raises(Exception, match="origin_tls"):
        LocalMetricConfig(origin_policy="explicit")
    with pytest.raises(Exception, match="only used"):
        LocalMetricConfig(origin_tls=[0, 0, 0])


# 10. a survey-scale offset collapses to a small, stable LOCAL_METRIC
def test_large_offset_becomes_small_local_coordinates():
    chain = build_frame_chain(SourceFrameConfig(mode="explicit_identity"), LocalMetricConfig(), BIG)
    local = chain.T_local_from_tls.apply(BIG)
    assert np.max(np.abs(local)) < 60.0
    assert np.max(np.abs(BIG)) > 1e6
