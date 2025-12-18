"""
Tests for the tools module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from f5_tts.tools.check_vocab import find_missing_tokens, load_vocab, merge_vocabularies, save_vocab
from f5_tts.tools.prepare_metadata import get_audio_duration


class TestVocabUtils:
    """Tests for vocabulary utilities."""

    def test_find_missing_tokens_empty(self):
        """Test finding missing tokens with identical vocabs."""
        pretrained = ["a", "b", "c"]
        dataset = ["a", "b", "c"]
        missing = find_missing_tokens(pretrained, dataset)
        assert missing == []

    def test_find_missing_tokens_some_missing(self):
        """Test finding missing tokens with some missing."""
        pretrained = ["a", "b", "c"]
        dataset = ["a", "b", "c", "d", "e"]
        missing = find_missing_tokens(pretrained, dataset)
        assert set(missing) == {"d", "e"}

    def test_find_missing_tokens_all_missing(self):
        """Test finding missing tokens with all missing."""
        pretrained = ["a", "b", "c"]
        dataset = ["x", "y", "z"]
        missing = find_missing_tokens(pretrained, dataset)
        assert set(missing) == {"x", "y", "z"}

    def test_merge_vocabularies(self):
        """Test merging vocabularies."""
        pretrained = ["a", "b", "c"]
        missing = ["d", "e"]
        merged = merge_vocabularies(pretrained, missing)
        assert merged == ["a", "b", "c", "d", "e"]

    def test_merge_vocabularies_empty_missing(self):
        """Test merging with no missing tokens."""
        pretrained = ["a", "b", "c"]
        missing = []
        merged = merge_vocabularies(pretrained, missing)
        assert merged == pretrained

    def test_save_and_load_vocab(self, temp_dir: Path):
        """Test saving and loading vocabulary."""
        vocab = ["token1", "token2", "token3"]
        vocab_path = temp_dir / "test_vocab.txt"

        save_vocab(str(vocab_path), vocab)
        loaded = load_vocab(str(vocab_path))

        assert loaded == vocab

    def test_load_vocab_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_vocab("/nonexistent/path/vocab.txt")

    def test_save_vocab_creates_file(self, temp_dir: Path):
        """Test that save_vocab creates the file."""
        vocab = ["a", "b", "c"]
        vocab_path = temp_dir / "new_vocab.txt"

        assert not vocab_path.exists()
        save_vocab(str(vocab_path), vocab)
        assert vocab_path.exists()


class TestPrepareMetadata:
    """Tests for metadata preparation utilities."""

    def test_get_audio_duration_function_exists(self):
        """Test that get_audio_duration function exists."""
        assert callable(get_audio_duration)
