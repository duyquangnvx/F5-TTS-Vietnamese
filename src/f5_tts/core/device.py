"""
Device detection and management utilities.

This module provides utilities for detecting and managing compute devices
(CUDA, XPU, MPS, CPU) and selecting appropriate data types.
"""

from __future__ import annotations

from typing import Optional

import torch


def get_device() -> str:
    """
    Detect and return the best available compute device.

    Returns:
        str: Device string ('cuda', 'xpu', 'mps', or 'cpu')

    Examples:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str | None = None) -> torch.dtype:
    """
    Get appropriate dtype for the specified device.

    Uses float16 on CUDA devices with compute capability >= 6.0,
    except for ZLUDA emulated devices. Falls back to float32 otherwise.

    Args:
        device: Device string. If None, auto-detects using get_device().

    Returns:
        torch.dtype: Recommended dtype for the device (float16 or float32)

    Examples:
        >>> dtype = get_dtype("cuda")
        >>> model = model.to(dtype)
    """
    if device is None:
        device = get_device()

    if "cuda" in device:
        try:
            props = torch.cuda.get_device_properties(device)
            device_name = torch.cuda.get_device_name()
            # Use float16 if compute capability >= 6.0 and not ZLUDA
            if props.major >= 6 and not device_name.endswith("[ZLUDA]"):
                return torch.float16
        except Exception:
            pass

    return torch.float32


def get_device_and_dtype() -> tuple[str, torch.dtype]:
    """
    Get both device and appropriate dtype in one call.

    Returns:
        tuple: (device string, dtype)

    Examples:
        >>> device, dtype = get_device_and_dtype()
        >>> model = model.to(device=device, dtype=dtype)
    """
    device = get_device()
    dtype = get_dtype(device)
    return device, dtype


def get_device_info(device: Optional[str] = None) -> dict:
    """
    Get detailed information about the compute device.

    Args:
        device: Device string. If None, auto-detects using get_device().

    Returns:
        dict: Device information including name, memory, compute capability, etc.
    """
    if device is None:
        device = get_device()

    info = {
        "device": device,
        "device_type": device.split(":")[0] if ":" in device else device,
        "device_name": "Unknown",
        "total_memory_gb": 0.0,
        "compute_capability": None,
        "dtype": str(get_dtype(device)),
    }

    if device.startswith("cuda"):
        try:
            device_idx = 0
            if ":" in device:
                device_idx = int(device.split(":")[1])

            props = torch.cuda.get_device_properties(device_idx)
            info["device_name"] = props.name
            info["total_memory_gb"] = props.total_memory / (1024**3)
            info["compute_capability"] = f"{props.major}.{props.minor}"
            info["multi_processor_count"] = props.multi_processor_count
            info["cuda_version"] = torch.version.cuda
        except Exception as e:
            info["error"] = str(e)

    elif device.startswith("xpu"):
        try:
            device_idx = 0
            if ":" in device:
                device_idx = int(device.split(":")[1])

            props = torch.xpu.get_device_properties(device_idx)
            info["device_name"] = props.name
            info["total_memory_gb"] = props.total_memory / (1024**3)
        except Exception as e:
            info["error"] = str(e)

    elif device == "mps":
        info["device_name"] = "Apple Silicon (MPS)"
        try:
            import psutil
            info["total_memory_gb"] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass

    elif device == "cpu":
        info["device_name"] = "CPU"
        try:
            import psutil
            info["total_memory_gb"] = psutil.virtual_memory().total / (1024**3)
            info["cpu_count"] = psutil.cpu_count()
        except ImportError:
            pass

    return info


def print_device_info(device: Optional[str] = None) -> str:
    """
    Print and return formatted device information.

    Args:
        device: Device string. If None, auto-detects using get_device().

    Returns:
        str: Formatted device information string.
    """
    info = get_device_info(device)

    lines = [
        "=" * 50,
        "Device Information",
        "=" * 50,
        f"  Device Type: {info['device_type'].upper()}",
        f"  Device Name: {info['device_name']}",
        f"  Memory: {info['total_memory_gb']:.2f} GB",
    ]

    if info.get("compute_capability"):
        lines.append(f"  Compute Capability: {info['compute_capability']}")

    if info.get("cuda_version"):
        lines.append(f"  CUDA Version: {info['cuda_version']}")

    if info.get("multi_processor_count"):
        lines.append(f"  Multiprocessors: {info['multi_processor_count']}")

    if info.get("cpu_count"):
        lines.append(f"  CPU Cores: {info['cpu_count']}")

    lines.append(f"  Dtype: {info['dtype']}")
    lines.append("=" * 50)

    output = "\n".join(lines)
    print(output)
    return output


def check_cuda_available() -> bool:
    """Check if CUDA is available and working."""
    if not torch.cuda.is_available():
        return False

    try:
        # Try to create a small tensor on GPU
        x = torch.zeros(1, device="cuda")
        del x
        return True
    except Exception:
        return False


def list_available_devices() -> list[dict]:
    """
    List all available compute devices.

    Returns:
        list: List of device info dictionaries for each available device.
    """
    devices = []

    # Check CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(get_device_info(f"cuda:{i}"))

    # Check XPU devices
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            devices.append(get_device_info(f"xpu:{i}"))

    # Check MPS
    if torch.backends.mps.is_available():
        devices.append(get_device_info("mps"))

    # Always include CPU
    devices.append(get_device_info("cpu"))

    return devices


def print_all_devices() -> None:
    """Print information about all available devices."""
    devices = list_available_devices()

    print("=" * 50)
    print("Available Compute Devices")
    print("=" * 50)

    for i, dev in enumerate(devices):
        marker = " [SELECTED]" if i == 0 and dev["device"] != "cpu" else ""
        print(f"\n[{i}] {dev['device'].upper()}{marker}")
        print(f"    Name: {dev['device_name']}")
        print(f"    Memory: {dev['total_memory_gb']:.2f} GB")
        if dev.get("compute_capability"):
            print(f"    Compute: {dev['compute_capability']}")

    print("\n" + "=" * 50)
