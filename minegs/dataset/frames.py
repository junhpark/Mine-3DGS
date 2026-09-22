"""Frame chain of a build (§5–§7, §27): SOURCE → TLS_GLOBAL → LOCAL_METRIC.

Two rules, both about not guessing:

* **SOURCE is not TLS_GLOBAL until someone says so.** ``resolve_tls_from_source`` only ever
  returns what the configuration declared. Identity is a declaration too (§5 A), and one that
  ``TLS_GLOBAL`` means *the survey's one metric reference frame*, not a geodetic CRS (§6).
* **LOCAL_METRIC is a translation of TLS_GLOBAL.** ``R = I``, scale 1, origin either declared
  or computed by a deterministic policy from the station positions. Re-running the same build
  yields the same transform, so two datasets from the same survey are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minegs.core.errors import ContractError
from minegs.core.frames import FLOAT32_SAFE_MAGNITUDE_M, SE3, Frame, check_float32_safe
from minegs.dataset.build_config import LocalMetricConfig, SourceFrameConfig

#: Rotation orthonormality / determinant tolerance for a declared transform. A declared
#: transform is typed by a person or produced by a registration tool; both round.
_RIGID_TOL = 1e-6
#: Round-trip error TLS -> LOCAL -> TLS that counts as "exact" (§27).
ROUNDTRIP_TOL_M = 1e-9


@dataclass(frozen=True)
class FrameChain:
    T_tls_from_source: SE3
    T_tls_from_local: SE3
    origin_policy: str
    source_mode: str

    @property
    def T_local_from_source(self) -> SE3:
        return self.T_tls_from_local.inverse() @ self.T_tls_from_source

    @property
    def T_local_from_tls(self) -> SE3:
        return self.T_tls_from_local.inverse()

    def to_record(self) -> dict:
        return {
            "source_mode": self.source_mode,
            "T_tls_from_source": self.T_tls_from_source.to_list(),
            "origin_policy": self.origin_policy,
            "T_tls_from_local": self.T_tls_from_local.to_list(),
            "local_origin_tls": self.T_tls_from_local.t.tolist(),
        }


def validate_rigid(M: np.ndarray | list, what: str) -> SE3:
    """A declared 4x4 must be finite, rigid and proper. Reflections are the classic silent
    killer here — a left-handed "transform" flips every camera and the overlay still looks
    plausible from a distance — so det = -1 is named explicitly."""
    A = np.asarray(M, dtype=np.float64)
    if A.shape != (4, 4):
        raise ContractError(f"{what}: expected a 4x4 matrix, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ContractError(f"{what}: matrix has non-finite entries")
    if not np.allclose(A[3], [0, 0, 0, 1], atol=1e-9):
        raise ContractError(f"{what}: last row must be [0 0 0 1], got {A[3].tolist()}")
    R = A[:3, :3]
    det = float(np.linalg.det(R))
    if det < 0:
        raise ContractError(
            f"{what}: rotation determinant is {det:.6f} — a reflection, not a rotation. A "
            "mirrored frame silently flips left and right in every camera."
        )
    orth = float(np.max(np.abs(R.T @ R - np.eye(3))))
    if orth > _RIGID_TOL or abs(det - 1.0) > _RIGID_TOL:
        raise ContractError(
            f"{what}: rotation is not orthonormal with det +1 (max |RᵀR−I| = {orth:.2e}, "
            f"det = {det:.6f}). Scale and shear are not allowed between SOURCE, TLS_GLOBAL and "
            "LOCAL_METRIC (§3: 1 unit = 1 m everywhere)."
        )
    return SE3.from_matrix(A)


def resolve_tls_from_source(cfg: SourceFrameConfig, source_frame: str = "SOURCE") -> SE3:
    """The declared SOURCE → TLS_GLOBAL transform, validated. Never inferred.

    ``source_frame`` is the frame the geometry actually arrives in, and there is exactly one
    value this door refuses: ``SFM_INTERNAL`` (Phase 3 AD-1). Both modes here are declarations
    and both return an SE(3), so promoting an independent reconstruction through them would
    assert that its arbitrary scale is already metres — which nobody measured. That promotion
    exists, and it is a measured Sim(3) with a registration artifact behind it.
    """
    if source_frame == Frame.SFM_INTERNAL.value:
        raise ContractError(
            "source_frame cannot promote SFM_INTERNAL geometry to TLS_GLOBAL. Both modes here "
            "are declarations and carry no scale, while an independent SfM reconstruction is "
            "arbitrary in scale as well as in origin: declaring it metric would make up the "
            "one number nobody measured. Register it first (`minegs eval register`), and build "
            "the dataset from the registration."
        )
    if cfg.mode == "explicit_identity":
        return SE3.identity()
    assert cfg.T_tls_from_source is not None  # pydantic guarantees it
    return validate_rigid(cfg.T_tls_from_source, "source_frame.T_tls_from_source")


def local_origin(cfg: LocalMetricConfig, station_positions_tls: np.ndarray) -> np.ndarray:
    """LOCAL_METRIC origin in TLS_GLOBAL, from the policy in the config (§7).

    ``station_centroid_rounded``: centroid of the registered station positions, rounded to
    ``rounding_m``. Rounding makes the transform readable and, more importantly, stable
    against float noise in the inputs. ``explicit``: exactly what was written.
    """
    if cfg.origin_policy == "explicit":
        origin = np.asarray(cfg.origin_tls, dtype=np.float64)
    else:
        p = np.asarray(station_positions_tls, dtype=np.float64).reshape(-1, 3)
        if len(p) == 0:
            raise ContractError("no station positions to derive a LOCAL_METRIC origin from")
        if not np.all(np.isfinite(p)):
            raise ContractError("station positions contain non-finite values")
        origin = np.round(p.mean(axis=0) / cfg.rounding_m) * cfg.rounding_m
    if not np.all(np.isfinite(origin)):
        raise ContractError("LOCAL_METRIC origin is not finite")
    return origin


def build_frame_chain(
    source_cfg: SourceFrameConfig,
    local_cfg: LocalMetricConfig,
    station_positions_tls: np.ndarray,
) -> FrameChain:
    T_tls_from_source = resolve_tls_from_source(source_cfg)
    origin = local_origin(local_cfg, station_positions_tls)
    T_tls_from_local = SE3.from_translation(origin)
    chain = FrameChain(
        T_tls_from_source, T_tls_from_local, local_cfg.origin_policy, source_cfg.mode
    )
    check_frame_chain(chain, station_positions_tls)
    return chain


def check_frame_chain(chain: FrameChain, sample_points_tls: np.ndarray | None = None) -> None:
    """§27 numerical sanity: finite, rigid, scale 1, round trip exact, LOCAL small enough."""
    for name, T in (
        ("T_tls_from_source", chain.T_tls_from_source),
        ("T_tls_from_local", chain.T_tls_from_local),
    ):
        validate_rigid(T.matrix(), name)
    if not np.allclose(chain.T_tls_from_local.R, np.eye(3), atol=1e-12):
        raise ContractError(
            "T_tls_from_local must be translation-only in the baseline (§7): no rotation, no "
            "PCA alignment, no normalisation"
        )
    pts = np.asarray(sample_points_tls, dtype=np.float64).reshape(-1, 3)
    if len(pts):
        back = chain.T_tls_from_local.apply(chain.T_local_from_tls.apply(pts))
        err = float(np.max(np.abs(back - pts)))
        if err > ROUNDTRIP_TOL_M:
            raise ContractError(f"TLS -> LOCAL -> TLS round trip error {err:.3e} m")
        local = chain.T_local_from_tls.apply(pts)
        mag = check_float32_safe(local, "LOCAL_METRIC station positions")
        if mag > FLOAT32_SAFE_MAGNITUDE_M / 5:
            raise ContractError(
                f"LOCAL_METRIC coordinates reach {mag:.0f} m from the origin; for a short "
                "tunnel that means the origin policy picked the wrong place"
            )
