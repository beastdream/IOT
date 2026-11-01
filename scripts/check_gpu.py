"""Report PyTorch and CUDA availability without starting model inference."""

import logging

from smart_helmet.inference.device import resolve_device


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        info = resolve_device("auto")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        logging.error("Could not inspect the inference device: %s", error)
        return 1

    print(f"PyTorch installed: {'yes' if info.torch_available else 'no'}")
    print(f"PyTorch version: {info.torch_version or 'unavailable'}")
    print(f"PyTorch CUDA build: {info.cuda_version or 'none (CPU-only build)'}")
    print(f"CUDA available: {'yes' if info.cuda_available else 'no'}")
    print(f"CUDA devices: {len(info.gpus)}")
    for gpu in info.gpus:
        print(f"GPU {gpu.index}: {gpu.name}")
    print(f"Auto selection: {info.device} ({info.message})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
