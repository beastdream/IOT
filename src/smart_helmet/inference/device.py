"""CUDA capability reporting and explicit inference-device selection."""

from dataclasses import dataclass
import importlib
from typing import Any


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested inference device cannot be used."""


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    device: str
    device_argument: str
    backend: str
    torch_available: bool
    torch_version: str | None
    cuda_version: str | None
    cuda_available: bool
    gpus: tuple[GPUInfo, ...]
    message: str


def _torch_module(torch_module: Any = None) -> Any:
    if torch_module is not None:
        return torch_module
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def resolve_device(requested: str = "auto", torch_module: Any = None) -> DeviceInfo:
    """Resolve auto/cpu/cuda[:index] without silently downgrading explicit CUDA."""
    request = str(requested).strip().lower()
    if request not in {"auto", "cpu", "cuda"} and not request.startswith("cuda:"):
        raise ValueError("Device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'.")

    torch = _torch_module(torch_module)
    if torch is None:
        if request.startswith("cuda"):
            raise DeviceUnavailableError(
                "CUDA was requested, but PyTorch is not installed. Install a PyTorch "
                "build compatible with your GPU, or use --device cpu."
            )
        return DeviceInfo(
            request, "cpu", "cpu", "cpu", False, None, None, False, (),
            "PyTorch is not installed; device reporting is limited to CPU.",
        )

    version = str(getattr(torch, "__version__", "unknown"))
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    cuda_available = bool(torch.cuda.is_available())
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpus = tuple(
        GPUInfo(index, str(torch.cuda.get_device_name(index)))
        for index in range(gpu_count)
    )
    cuda_available = cuda_available and bool(gpus)

    if request == "cpu":
        return DeviceInfo(
            request, "cpu", "cpu", "cpu", True, version, cuda_version,
            cuda_available, gpus, "CPU explicitly selected.",
        )
    if request == "auto":
        if cuda_available:
            return DeviceInfo(
                request, "cuda:0", "0", "cuda", True, version, cuda_version,
                True, gpus, "CUDA is available; selected GPU 0.",
            )
        return DeviceInfo(
            request, "cpu", "cpu", "cpu", True, version, cuda_version,
            False, gpus, "CUDA is unavailable; auto mode selected CPU.",
        )

    index = 0
    if request.startswith("cuda:"):
        value = request.removeprefix("cuda:")
        if not value.isdecimal():
            raise ValueError("CUDA device index must be a non-negative integer.")
        index = int(value)
    if not cuda_available or index >= len(gpus):
        raise DeviceUnavailableError(
            f"CUDA device {index} is unavailable (CUDA available: "
            f"{cuda_available}; GPUs found: {len(gpus)}). Use --device auto or "
            "--device cpu, and verify the NVIDIA driver and CUDA-enabled PyTorch."
        )
    return DeviceInfo(
        request, f"cuda:{index}", str(index), "cuda", True, version,
        cuda_version, True, gpus, f"CUDA GPU {index} explicitly selected.",
    )


def verify_inference_device(model: Any, selection: DeviceInfo) -> tuple[str, bool]:
    """Read the loaded Ultralytics backend device and reject a false CUDA claim."""
    predictor = getattr(model, "predictor", None)
    backend = getattr(predictor, "model", None)
    actual = getattr(backend, "device", None)
    if actual is None:
        raw_model = getattr(backend, "model", None)
        parameters = getattr(raw_model, "parameters", None)
        if callable(parameters):
            actual = next(parameters(), None)
            actual = getattr(actual, "device", None)
    if actual is None:
        if selection.backend == "cuda":
            raise DeviceUnavailableError(
                "CUDA was selected, but the model backend did not expose its "
                "active device; GPU inference cannot be confirmed."
            )
        return selection.device, False

    actual_device = str(actual)
    actual_backend = "cuda" if actual_device.startswith("cuda") else actual_device
    if actual_backend != selection.backend:
        raise DeviceUnavailableError(
            f"Model inference is running on {actual_device}, not the requested "
            f"{selection.device}."
        )
    if selection.backend == "cuda" and actual_device != selection.device:
        raise DeviceUnavailableError(
            f"Model is running on {actual_device}, not the requested {selection.device}."
        )
    return actual_device, True
