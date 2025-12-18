"""
Core shared components for F5-TTS.

This module contains shared utilities and components used across
different parts of the F5-TTS system.
"""

from f5_tts.core.device import (
    get_device,
    get_dtype,
    get_device_and_dtype,
    get_device_info,
    print_device_info,
    check_cuda_available,
    list_available_devices,
    print_all_devices,
)
from f5_tts.core.embeddings import (
    TextEmbedding,
    MMDiTTextEmbedding,
    InputEmbedding,
    AudioEmbedding,
)

__all__ = [
    # Device utilities
    "get_device",
    "get_dtype",
    "get_device_and_dtype",
    "get_device_info",
    "print_device_info",
    "check_cuda_available",
    "list_available_devices",
    "print_all_devices",
    # Embeddings
    "TextEmbedding",
    "MMDiTTextEmbedding",
    "InputEmbedding",
    "AudioEmbedding",
]
