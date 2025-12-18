"""
Tests for the core module (device and embeddings).
"""

from __future__ import annotations

import pytest
import torch

from f5_tts.core import get_device, get_dtype


class TestDeviceDetection:
    """Tests for device detection utilities."""

    def test_get_device_returns_string(self):
        """Test that get_device returns a string."""
        device = get_device()
        assert isinstance(device, str)

    def test_get_device_valid_values(self):
        """Test that get_device returns a valid device string."""
        device = get_device()
        assert device in ["cuda", "xpu", "mps", "cpu"]

    def test_get_device_consistency(self):
        """Test that get_device returns consistent results."""
        device1 = get_device()
        device2 = get_device()
        assert device1 == device2

    def test_get_dtype_returns_torch_dtype(self):
        """Test that get_dtype returns a torch dtype."""
        dtype = get_dtype()
        assert dtype in [torch.float16, torch.float32]

    def test_get_dtype_with_device(self):
        """Test get_dtype with explicit device."""
        dtype_cpu = get_dtype("cpu")
        assert dtype_cpu == torch.float32

    def test_get_dtype_with_cuda(self):
        """Test get_dtype with cuda device."""
        if torch.cuda.is_available():
            dtype_cuda = get_dtype("cuda")
            # Should be float16 for capable GPUs
            assert dtype_cuda in [torch.float16, torch.float32]
        else:
            pytest.skip("CUDA not available")


class TestEmbeddings:
    """Tests for embedding classes."""

    def test_text_embedding_import(self):
        """Test that TextEmbedding can be imported."""
        from f5_tts.core.embeddings import TextEmbedding
        assert TextEmbedding is not None

    def test_input_embedding_import(self):
        """Test that InputEmbedding can be imported."""
        from f5_tts.core.embeddings import InputEmbedding
        assert InputEmbedding is not None

    def test_mmdit_text_embedding_import(self):
        """Test that MMDiTTextEmbedding can be imported."""
        from f5_tts.core.embeddings import MMDiTTextEmbedding
        assert MMDiTTextEmbedding is not None

    def test_audio_embedding_import(self):
        """Test that AudioEmbedding can be imported."""
        from f5_tts.core.embeddings import AudioEmbedding
        assert AudioEmbedding is not None

    @pytest.mark.parametrize("dim", [256, 512, 1024])
    def test_text_embedding_initialization(self, dim: int):
        """Test TextEmbedding initialization with various dimensions."""
        from f5_tts.core.embeddings import TextEmbedding

        num_embeds = 256
        embed = TextEmbedding(dim=dim, num_embeds=num_embeds)
        assert embed is not None

    @pytest.mark.parametrize("mel_dim", [80, 100, 128])
    def test_input_embedding_initialization(self, mel_dim: int):
        """Test InputEmbedding initialization with various mel dimensions."""
        from f5_tts.core.embeddings import InputEmbedding

        dim = 512
        embed = InputEmbedding(mel_dim=mel_dim, dim=dim)
        assert embed is not None

    def test_text_embedding_forward(self):
        """Test TextEmbedding forward pass."""
        from f5_tts.core.embeddings import TextEmbedding

        dim = 256
        num_embeds = 100
        embed = TextEmbedding(dim=dim, num_embeds=num_embeds)

        batch_size = 2
        seq_len = 10
        # Input should be indices
        x = torch.randint(0, num_embeds, (batch_size, seq_len))

        output = embed(x, drop=False)
        assert output.shape == (batch_size, seq_len, dim)

    def test_input_embedding_forward(self):
        """Test InputEmbedding forward pass."""
        from f5_tts.core.embeddings import InputEmbedding

        mel_dim = 100
        dim = 512
        embed = InputEmbedding(mel_dim=mel_dim, dim=dim)

        batch_size = 2
        seq_len = 50
        x = torch.randn(batch_size, seq_len, mel_dim)
        cond = torch.randn(batch_size, seq_len, mel_dim)

        output = embed(x, cond)
        assert output.shape == (batch_size, seq_len, dim)
