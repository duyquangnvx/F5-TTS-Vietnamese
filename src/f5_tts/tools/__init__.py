"""
F5-TTS Tools Package.

This package contains utility scripts for data preparation,
vocabulary management, and model manipulation.
"""

from f5_tts.tools.prepare_metadata import main as prepare_metadata
from f5_tts.tools.convert_sr import main as convert_sr
from f5_tts.tools.check_vocab import main as check_vocab
from f5_tts.tools.extend_embeddings import main as extend_embeddings

__all__ = [
    "prepare_metadata",
    "convert_sr",
    "check_vocab",
    "extend_embeddings",
]
