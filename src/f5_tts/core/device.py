"""
Device detection and management utilities.

This module provides utilities for detecting and managing compute devices
(CUDA, XPU, MPS, CPU) and selecting appropriate data types.
"""

from __future__ import annotations

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
