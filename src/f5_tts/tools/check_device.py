#!/usr/bin/env python3
"""
Tool to check and display available compute devices.

This script helps users verify their GPU setup and see what devices
are available for F5-TTS inference and training.

Usage:
    python -m f5_tts.tools.check_device
    # or
    f5-tts-check-device (if installed)
"""

import argparse

from f5_tts.core.device import (
    get_device,
    get_dtype,
    print_device_info,
    print_all_devices,
    check_cuda_available,
)


def main():
    parser = argparse.ArgumentParser(
        description="Check available compute devices for F5-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m f5_tts.tools.check_device
    python -m f5_tts.tools.check_device --all
    python -m f5_tts.tools.check_device --test
        """,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all available devices instead of just the selected one",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test CUDA by creating a tensor on GPU",
    )
    args = parser.parse_args()

    if args.all:
        print_all_devices()
    else:
        print_device_info()

    if args.test:
        print("\nTesting CUDA availability...")
        if check_cuda_available():
            print("[OK] CUDA is working correctly!")
            print(f"    Default device: {get_device()}")
            print(f"    Default dtype: {get_dtype()}")
        else:
            print("[WARNING] CUDA is not available or not working")
            print("         Falling back to CPU for computation")


if __name__ == "__main__":
    main()
