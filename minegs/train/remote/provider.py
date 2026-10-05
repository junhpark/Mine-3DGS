"""The provider seam (Phase 6 §7): create, inspect and terminate one pod.

Only ``SdkRunPodClient`` imports the ``runpod`` SDK (pinned to the 1.12 line, whose
``create_pod`` / ``get_pod`` / ``terminate_pod`` signatures were read from source). The rest of
MineGS sees ``PodSpec`` and ``PodInfo``; the SDK's dictionaries stop here. Tests use a fake client
behind the same three methods — the network API is the only thing substituted.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from minegs.core.errors import ContractError
from minegs.train.remote.secrets import API_KEY_ENV, redact


class ProviderError(ContractError):
    """The provider refused or failed a request (message already redacted)."""


class ProviderAllocationError(ProviderError):
    """No machine with the requested GPU could be allocated. Nothing was created."""


class ProviderUnavailableError(ProviderError):
    """The provider could not be asked (timeout, network). Says nothing about the pod."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PodSpec(_Strict):
    name: str
    image: str
    gpu_type_id: str
    gpu_count: int = 1
    network_volume_id: str
    volume_mount_path: str
    container_disk_gb: int
    docker_args: str
    env: dict[str, str] = Field(default_factory=dict)
    cloud_type: str = "ALL"
    allowed_cuda_versions: list[str] | None = None


class PodInfo(_Strict):
    pod_id: str
    #: RunPod ``desiredStatus`` (RUNNING, EXITED, TERMINATED, ...): the pod's lifecycle, nothing
    #: about the trainer.
    desired_status: str | None = None
    gpu_display_name: str | None = None
    gpu_count: int | None = None
    image: str | None = None
    uptime_seconds: int | None = None
    #: The pod's hourly rate as the provider states it — a rate, never a billed total.
    cost_per_hr: float | None = None
    last_status_change: str | None = None


#: Lifecycle states after which the pod's process will not write anything more.
ENDED_STATES = frozenset({"EXITED", "TERMINATED", "DEAD", "STOPPED"})


class RunPodClient:
    """The three calls Phase 6 needs."""

    def create_pod(self, spec: PodSpec) -> PodInfo:  # pragma: no cover - interface
        raise NotImplementedError

    def get_pod(self, pod_id: str) -> PodInfo | None:  # pragma: no cover - interface
        raise NotImplementedError

    def terminate_pod(self, pod_id: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError


def require_api_key(environ: dict[str, str] | None = None) -> str:
    """The name of the credential, after checking it is there. The value is not returned."""
    env = os.environ if environ is None else environ
    if not (env.get(API_KEY_ENV) or "").strip():
        raise ContractError(
            f"{API_KEY_ENV} is not set; RunPod runs need an API key in the environment (it is "
            "never read from a config file and never recorded)"
        )
    return API_KEY_ENV


def _info(d: dict[str, Any] | None) -> PodInfo | None:
    if not d:
        return None
    machine = d.get("machine") or {}
    return PodInfo(
        pod_id=str(d.get("id")),
        desired_status=d.get("desiredStatus"),
        gpu_display_name=machine.get("gpuDisplayName") if isinstance(machine, dict) else None,
        gpu_count=d.get("gpuCount"),
        image=d.get("imageName"),
        uptime_seconds=d.get("uptimeSeconds"),
        cost_per_hr=d.get("costPerHr"),
        last_status_change=d.get("lastStatusChange"),
    )


class SdkRunPodClient(RunPodClient):
    """``runpod`` 1.12.x. The key goes to ``runpod.api_key`` only, just before a call."""

    def __init__(self) -> None:
        require_api_key()
        try:
            import runpod  # noqa: F401
        except ImportError as e:
            raise ContractError(
                "the runpod SDK is not installed: pip install 'minegs[runpod]' (pinned 1.12.x)"
            ) from e

    def _sdk(self):
        import runpod

        runpod.api_key = os.environ[API_KEY_ENV]
        return runpod

    def _call(self, what: str, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (TimeoutError, ConnectionError, OSError) as e:
            raise ProviderUnavailableError(f"RunPod {what}: {redact(e)}") from None
        except Exception as e:  # SDK errors (QueryError, AuthenticationError, ValueError, ...)
            name = type(e).__name__
            text = redact(e)
            if "instances" in text.lower() and "available" in text.lower():
                raise ProviderAllocationError(f"RunPod {what}: {name}: {text}") from None
            raise ProviderError(f"RunPod {what}: {name}: {text}") from None

    def create_pod(self, spec: PodSpec) -> PodInfo:
        sdk = self._sdk()
        d = self._call(
            "create_pod",
            sdk.create_pod,
            name=spec.name,
            image_name=spec.image,
            gpu_type_id=spec.gpu_type_id,
            cloud_type=spec.cloud_type,
            support_public_ip=False,
            start_ssh=False,
            gpu_count=spec.gpu_count,
            container_disk_in_gb=spec.container_disk_gb,
            docker_args=spec.docker_args,
            volume_mount_path=spec.volume_mount_path,
            env=dict(spec.env),
            network_volume_id=spec.network_volume_id,
            allowed_cuda_versions=spec.allowed_cuda_versions,
        )
        info = _info(d)
        if info is None:
            raise ProviderError("RunPod create_pod returned no pod")
        return info

    def get_pod(self, pod_id: str) -> PodInfo | None:
        sdk = self._sdk()
        return _info(self._call("get_pod", sdk.get_pod, pod_id))

    def terminate_pod(self, pod_id: str) -> None:
        sdk = self._sdk()
        self._call("terminate_pod", sdk.terminate_pod, pod_id)


__all__ = [
    "ENDED_STATES",
    "PodInfo",
    "PodSpec",
    "ProviderAllocationError",
    "ProviderError",
    "ProviderUnavailableError",
    "RunPodClient",
    "SdkRunPodClient",
    "require_api_key",
]
