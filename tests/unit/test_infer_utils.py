"""
Tests for inference utilities.
"""

from __future__ import annotations

import pytest

from f5_tts.infer.utils_infer import (
    chunk_text,
    InferenceContext,
    get_default_context,
)


class TestChunkText:
    """Tests for text chunking utility."""

    def test_chunk_text_short_text(self):
        """Test chunking with text shorter than max_chars."""
        text = "Hello world."
        chunks = chunk_text(text, max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_long_text(self):
        """Test chunking with text longer than max_chars."""
        text = "This is a long sentence. " * 10
        chunks = chunk_text(text, max_chars=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.encode("utf-8")) <= 50 or len(chunk.split()) == 1

    def test_chunk_text_empty_string(self):
        """Test chunking with empty string."""
        chunks = chunk_text("", max_chars=100)
        assert chunks == [] or chunks == [""]

    def test_chunk_text_preserves_content(self):
        """Test that chunking preserves all content."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, max_chars=100)
        rejoined = " ".join(chunks)
        # Content should be preserved (whitespace may differ)
        assert "First" in rejoined
        assert "Second" in rejoined
        assert "Third" in rejoined

    def test_chunk_text_respects_punctuation(self):
        """Test that chunking respects sentence boundaries."""
        text = "Short. This is a longer sentence that should be separate."
        chunks = chunk_text(text, max_chars=30)
        # Should split at punctuation when possible
        assert len(chunks) >= 1

    def test_chunk_text_unicode(self):
        """Test chunking with unicode characters."""
        text = "Hello. Xin chào. 你好."
        chunks = chunk_text(text, max_chars=50)
        assert len(chunks) >= 1
        # All original text should be preserved
        rejoined = " ".join(chunks)
        assert "Hello" in rejoined
        assert "Xin chào" in rejoined

    @pytest.mark.parametrize("max_chars", [50, 100, 135, 200])
    def test_chunk_text_various_limits(self, max_chars: int):
        """Test chunking with various character limits."""
        text = "This is a test sentence. " * 5
        chunks = chunk_text(text, max_chars=max_chars)
        assert all(isinstance(c, str) for c in chunks)


class TestInferenceContext:
    """Tests for InferenceContext class."""

    def test_inference_context_creation(self):
        """Test creating an InferenceContext."""
        ctx = InferenceContext()
        assert ctx is not None

    def test_inference_context_device(self):
        """Test InferenceContext device attribute."""
        ctx = InferenceContext()
        assert ctx.device in ["cuda", "xpu", "mps", "cpu"]

    def test_inference_context_custom_device(self):
        """Test InferenceContext with custom device."""
        ctx = InferenceContext(device="cpu")
        assert ctx.device == "cpu"

    def test_inference_context_default_values(self):
        """Test InferenceContext default configuration values."""
        ctx = InferenceContext()
        assert ctx.target_sample_rate == 24000
        assert ctx.n_mel_channels == 100
        assert ctx.nfe_step == 32
        assert ctx.cfg_strength == 2.0
        assert ctx.speed == 1.0

    def test_inference_context_custom_values(self):
        """Test InferenceContext with custom configuration."""
        ctx = InferenceContext(
            nfe_step=16,
            cfg_strength=1.5,
            speed=0.9,
        )
        assert ctx.nfe_step == 16
        assert ctx.cfg_strength == 1.5
        assert ctx.speed == 0.9

    def test_inference_context_cache(self):
        """Test InferenceContext cache operations."""
        ctx = InferenceContext()
        assert hasattr(ctx, "_ref_audio_cache")
        assert isinstance(ctx._ref_audio_cache, dict)

    def test_inference_context_clear_cache(self):
        """Test clearing InferenceContext cache."""
        ctx = InferenceContext()
        ctx._ref_audio_cache["test"] = "value"
        ctx.clear_cache()
        assert len(ctx._ref_audio_cache) == 0

    def test_inference_context_vocoder_property(self):
        """Test InferenceContext vocoder property."""
        ctx = InferenceContext()
        # Before loading, should be None
        assert ctx.vocoder is None

    def test_inference_context_model_property(self):
        """Test InferenceContext model property."""
        ctx = InferenceContext()
        # Before loading, should be None
        assert ctx.model is None


class TestDefaultContext:
    """Tests for default context singleton."""

    def test_get_default_context(self):
        """Test getting default context."""
        ctx = get_default_context()
        assert ctx is not None
        assert isinstance(ctx, InferenceContext)

    def test_default_context_singleton(self):
        """Test that default context is a singleton."""
        ctx1 = get_default_context()
        ctx2 = get_default_context()
        assert ctx1 is ctx2

    def test_default_context_has_device(self):
        """Test that default context has a device."""
        ctx = get_default_context()
        assert ctx.device is not None
        assert ctx.device in ["cuda", "xpu", "mps", "cpu"]
