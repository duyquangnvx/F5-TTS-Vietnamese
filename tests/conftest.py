"""
Pytest configuration and fixtures for F5-TTS tests.

This module provides shared fixtures and configuration for all tests.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def src_dir(project_root: Path) -> Path:
    """Return the source directory."""
    return project_root / "src" / "f5_tts"


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="function")
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file for tests."""
    file_path = temp_dir / "test_file.txt"
    file_path.touch()
    yield file_path


@pytest.fixture(scope="session")
def sample_vocab() -> list[str]:
    """Return a sample vocabulary list for testing."""
    return [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z", " ", ".", ",", "!",
    ]


@pytest.fixture(scope="function")
def vocab_file(temp_dir: Path, sample_vocab: list[str]) -> Path:
    """Create a temporary vocabulary file."""
    vocab_path = temp_dir / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_vocab))
    return vocab_path


@pytest.fixture(scope="session")
def device() -> str:
    """Return the appropriate device for testing."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="session")
def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
