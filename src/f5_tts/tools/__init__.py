"""
F5-TTS Tools Package.

This package contains utility scripts for data preparation,
vocabulary management, model manipulation, and device diagnostics.

Tools are available as CLI commands (f5-tts-*) or can be imported:
    from f5_tts.tools import check_device
    check_device()
"""

__all__ = [
    "prepare_metadata",
    "convert_sr",
    "check_vocab",
    "extend_embeddings",
    "check_device",
    "download_models",
    "check_pytorch",
]


def __getattr__(name: str):
    """Lazy import tools to avoid circular import warnings when running as modules."""
    if name == "prepare_metadata":
        from f5_tts.tools.prepare_metadata import main
        return main
    elif name == "convert_sr":
        from f5_tts.tools.convert_sr import main
        return main
    elif name == "check_vocab":
        from f5_tts.tools.check_vocab import main
        return main
    elif name == "extend_embeddings":
        from f5_tts.tools.extend_embeddings import main
        return main
    elif name == "check_device":
        from f5_tts.tools.check_device import main
        return main
    elif name == "download_models":
        from f5_tts.tools.download_models import main
        return main
    elif name == "check_pytorch":
        from f5_tts.tools.check_pytorch import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
