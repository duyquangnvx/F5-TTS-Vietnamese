#!/usr/bin/env python3
"""
PyTorch installation validation tool.

This tool helps users:
1. Check if PyTorch is correctly installed with CUDA
2. Detect GPU and recommend appropriate PyTorch version
3. Warn if CPU-only PyTorch is on a GPU machine

Usage:
    python -m f5_tts.tools.check_pytorch
    python -m f5_tts.tools.check_pytorch --recommend
    python -m f5_tts.tools.check_pytorch --quiet
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Optional

# GPU series to CUDA version mapping
GPU_CUDA_MAPPING = {
    "RTX 50": {"cuda": "12.8", "nightly": True, "arch": "Blackwell"},
    "RTX 40": {"cuda": "12.4", "nightly": False, "arch": "Ada Lovelace"},
    "RTX 30": {"cuda": "12.1", "nightly": False, "arch": "Ampere"},
    "RTX A": {"cuda": "12.1", "nightly": False, "arch": "Ampere"},
    "RTX 20": {"cuda": "11.8", "nightly": False, "arch": "Turing"},
    "GTX 16": {"cuda": "11.8", "nightly": False, "arch": "Turing"},
    "GTX 10": {"cuda": "11.8", "nightly": False, "arch": "Pascal"},
}

# CUDA version to PyTorch index URL
CUDA_INDEX_URLS = {
    "12.8": "https://download.pytorch.org/whl/nightly/cu128",
    "12.4": "https://download.pytorch.org/whl/cu124",
    "12.1": "https://download.pytorch.org/whl/cu121",
    "11.8": "https://download.pytorch.org/whl/cu118",
    "cpu": None,  # Use default PyPI
}


def get_nvidia_smi_info() -> Optional[dict]:
    """
    Get GPU information from nvidia-smi without importing torch.

    Returns:
        dict with gpu_name, cuda_version, driver_version, gpu_series
        or None if nvidia-smi is not available
    """
    try:
        # Check if nvidia-smi exists
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        gpu_name = result.stdout.strip()

        # Get CUDA and driver version
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        cuda_version = ""
        driver_version = ""

        import re

        cuda_match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
        driver_match = re.search(r"Driver Version:\s*(\d+\.\d+)", result.stdout)

        if cuda_match:
            cuda_version = cuda_match.group(1)
        if driver_match:
            driver_version = driver_match.group(1)

        # Detect GPU series
        gpu_series = "Unknown"
        for series in GPU_CUDA_MAPPING:
            if series in gpu_name:
                gpu_series = series
                break

        return {
            "gpu_name": gpu_name,
            "cuda_version": cuda_version,
            "driver_version": driver_version,
            "gpu_series": gpu_series,
        }

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def get_pytorch_status() -> dict:
    """
    Get current PyTorch installation status.

    Returns:
        dict with installed, version, cuda_available, cuda_version, is_cpu_only
    """
    result = {
        "installed": False,
        "version": "",
        "cuda_available": False,
        "cuda_version": "",
        "is_cpu_only": False,
    }

    try:
        import torch

        result["installed"] = True
        result["version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()

        if result["cuda_available"]:
            result["cuda_version"] = torch.version.cuda or ""
        else:
            # Check if it's CPU-only build (version ends with +cpu)
            result["is_cpu_only"] = "+cpu" in torch.__version__

    except ImportError:
        pass

    return result


def get_recommended_cuda(gpu_series: str) -> dict:
    """Get recommended CUDA version for a GPU series."""
    if gpu_series in GPU_CUDA_MAPPING:
        return GPU_CUDA_MAPPING[gpu_series]
    return {"cuda": "12.1", "nightly": False, "arch": "Unknown"}


def get_install_command(cuda_version: str) -> str:
    """Get pip install command for a CUDA version."""
    packages = "torch torchvision torchaudio"

    if cuda_version == "cpu":
        return f"pip install {packages}"

    if cuda_version == "12.8":
        return f"pip install --pre {packages} --index-url {CUDA_INDEX_URLS[cuda_version]}"

    if cuda_version in CUDA_INDEX_URLS:
        return f"pip install {packages} --index-url {CUDA_INDEX_URLS[cuda_version]}"

    # Default to 12.1
    return f"pip install {packages} --index-url {CUDA_INDEX_URLS['12.1']}"


def print_status(pytorch_status: dict, gpu_info: Optional[dict]) -> None:
    """Print current status."""
    print("=" * 60)
    print("  PyTorch Installation Check")
    print("=" * 60)

    # PyTorch status
    if pytorch_status["installed"]:
        print(f"  PyTorch Installed: Yes (v{pytorch_status['version']})")
        cuda_status = "Yes" if pytorch_status["cuda_available"] else "No (CPU-only build)"
        print(f"  CUDA Available: {cuda_status}")
        if pytorch_status["cuda_version"]:
            print(f"  PyTorch CUDA Version: {pytorch_status['cuda_version']}")
    else:
        print("  PyTorch Installed: No")

    # GPU status
    if gpu_info:
        print(f"  GPU Detected: Yes ({gpu_info['gpu_name']})")
        print(f"  GPU Series: {gpu_info['gpu_series']}")
        print(f"  Driver CUDA Version: {gpu_info['cuda_version']}")
    else:
        print("  GPU Detected: No")

    print("=" * 60)


def validate_installation() -> tuple[bool, str]:
    """
    Validate current PyTorch installation.

    Returns:
        (is_valid, message)
    """
    pytorch_status = get_pytorch_status()
    gpu_info = get_nvidia_smi_info()

    # Case 1: PyTorch not installed
    if not pytorch_status["installed"]:
        if gpu_info:
            recommended = get_recommended_cuda(gpu_info["gpu_series"])
            cmd = get_install_command(recommended["cuda"])
            return False, f"PyTorch is not installed.\n\nRecommended command:\n  {cmd}"
        else:
            cmd = get_install_command("cpu")
            return False, f"PyTorch is not installed.\n\nRecommended command:\n  {cmd}"

    # Case 2: PyTorch installed with CUDA working
    if pytorch_status["cuda_available"]:
        return True, "PyTorch is correctly configured for GPU!"

    # Case 3: CPU-only PyTorch on a GPU machine
    if gpu_info and not pytorch_status["cuda_available"]:
        recommended = get_recommended_cuda(gpu_info["gpu_series"])
        uninstall_cmd = "pip uninstall torch torchvision torchaudio -y"
        install_cmd = get_install_command(recommended["cuda"])

        msg = (
            f"CPU-only PyTorch detected on a GPU machine!\n\n"
            f"Your GPU ({gpu_info['gpu_name']}) requires CUDA {recommended['cuda']}.\n\n"
            f"To fix, run:\n"
            f"  {uninstall_cmd}\n"
            f"  {install_cmd}\n\n"
            f"Or use the setup script:\n"
            f"  .\\scripts\\setup_pytorch.ps1"
        )
        return False, msg

    # Case 4: CPU-only PyTorch, no GPU (this is fine)
    if not gpu_info and not pytorch_status["cuda_available"]:
        return True, "PyTorch (CPU-only) is correctly configured for your system."

    return True, "PyTorch installation looks OK."


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check PyTorch installation and GPU compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m f5_tts.tools.check_pytorch           # Full status check
  python -m f5_tts.tools.check_pytorch --recommend   # Show install command
  python -m f5_tts.tools.check_pytorch --quiet   # Only show warnings
        """,
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Show recommended PyTorch install command for your GPU",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only show output if there's an issue",
    )

    args = parser.parse_args()

    pytorch_status = get_pytorch_status()
    gpu_info = get_nvidia_smi_info()

    if args.recommend:
        # Just show recommended command
        if gpu_info:
            recommended = get_recommended_cuda(gpu_info["gpu_series"])
            print(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_series']})")
            print(f"Recommended CUDA: {recommended['cuda']}")
            print()
            print("Install command:")
            print(f"  {get_install_command(recommended['cuda'])}")
        else:
            print("No GPU detected. Use CPU-only PyTorch:")
            print(f"  {get_install_command('cpu')}")
        return

    # Validate installation
    is_valid, message = validate_installation()

    if args.quiet:
        if not is_valid:
            print()
            print("[WARNING] " + message.split("\n")[0])
            print()
            print(message)
            sys.exit(1)
        return

    # Full output
    print()
    print_status(pytorch_status, gpu_info)
    print()

    if is_valid:
        print(f"[OK] {message}")
    else:
        print(f"[WARNING] {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
