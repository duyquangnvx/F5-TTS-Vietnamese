"""
F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

Vietnamese fine-tuning version.

This package provides text-to-speech synthesis using flow matching,
with support for multiple languages including Vietnamese.
"""

__version__ = "1.0.1"

from f5_tts.api import F5TTS
from f5_tts.constants import (
    TARGET_SAMPLE_RATE,
    N_MEL_CHANNELS,
    HOP_LENGTH,
    WIN_LENGTH,
    N_FFT,
    DEFAULT_NFE_STEP,
    DEFAULT_CFG_STRENGTH,
    MODEL_CONFIGS,
    PRETRAINED_MODELS,
    EXPERIMENT_NAMES,
)

__all__ = [
    # Main API
    "F5TTS",
    # Version
    "__version__",
    # Constants
    "TARGET_SAMPLE_RATE",
    "N_MEL_CHANNELS",
    "HOP_LENGTH",
    "WIN_LENGTH",
    "N_FFT",
    "DEFAULT_NFE_STEP",
    "DEFAULT_CFG_STRENGTH",
    "MODEL_CONFIGS",
    "PRETRAINED_MODELS",
    "EXPERIMENT_NAMES",
]
