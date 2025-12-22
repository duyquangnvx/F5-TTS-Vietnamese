"""
F5-TTS Tools Package.

This package contains utility scripts for data preparation,
vocabulary management, model manipulation, and device diagnostics.
"""

from f5_tts.tools.prepare_metadata import main as prepare_metadata
from f5_tts.tools.convert_sr import main as convert_sr
from f5_tts.tools.check_vocab import main as check_vocab
from f5_tts.tools.extend_embeddings import main as extend_embeddings
from f5_tts.tools.check_device import main as check_device
from f5_tts.tools.download_models import main as download_models
from f5_tts.tools.check_pytorch import main as check_pytorch

__all__ = [
    "prepare_metadata",
    "convert_sr",
    "check_vocab",
    "extend_embeddings",
    "check_device",
    "download_models",
    "check_pytorch",
]
