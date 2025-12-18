"""
Core shared components for F5-TTS.

This module contains shared utilities and components used across
different parts of the F5-TTS system.
"""

from f5_tts.core.device import get_device, get_dtype
from f5_tts.core.embeddings import (
    TextEmbedding,
    MMDiTTextEmbedding,
    InputEmbedding,
    AudioEmbedding,
)

__all__ = [
    "get_device",
    "get_dtype",
    "TextEmbedding",
    "MMDiTTextEmbedding",
    "InputEmbedding",
    "AudioEmbedding",
]
