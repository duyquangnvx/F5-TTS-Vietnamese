"""
Shared embedding modules used across different backbone architectures.

This module consolidates the TextEmbedding and InputEmbedding classes
that were previously duplicated in dit.py, unett.py, and mmdit.py.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from f5_tts.model.modules import (
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    precompute_freqs_cis,
    get_pos_embed_indices,
)
from f5_tts.constants import PRECOMPUTE_MAX_POS


class TextEmbedding(nn.Module):
    """
    Text embedding module with optional convolutional layers.

    Used by DiT and UNetT backbones. Supports optional extra modeling
    with ConvNeXtV2 blocks and positional embeddings.

    Args:
        text_num_embeds: Number of text embeddings (vocabulary size)
        text_dim: Dimension of text embeddings
        mask_padding: Whether to mask filler and batch padding tokens
        conv_layers: Number of ConvNeXtV2 blocks for extra modeling (0 to disable)
        conv_mult: Multiplier for ConvNeXtV2 intermediate dimension
    """

    def __init__(
        self,
        text_num_embeds: int,
        text_dim: int,
        mask_padding: bool = True,
        conv_layers: int = 0,
        conv_mult: int = 2,
    ):
        super().__init__()
        # Use 0 as filler token, so we need +1 embeddings
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.mask_padding = mask_padding

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = PRECOMPUTE_MAX_POS
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(text_dim, self.precompute_max_pos),
                persistent=False,
            )
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(
        self,
        text: torch.Tensor,  # int["b nt"]
        seq_len: int,
        drop_text: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for text embedding.

        Args:
            text: Input text token indices [batch, text_len]
            seq_len: Target sequence length to pad/truncate to
            drop_text: Whether to drop text (for classifier-free guidance)

        Returns:
            Embedded text tensor [batch, seq_len, text_dim]
        """
        # Use 0 as filler token. Preprocess of batch pad -1, see list_str_to_idx()
        text = text + 1
        # Curtail if character tokens are more than the mel spec tokens
        text = text[:, :seq_len]
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # Possible extra modeling
        if self.extra_modeling:
            # Sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long, device=text.device)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # ConvNeXtV2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        return text


class MMDiTTextEmbedding(nn.Module):
    """
    Text embedding module for MMDiT backbone.

    Simpler variant without conv layers, with built-in positional embedding.
    Uses a smaller precompute_max_pos (1024 vs 4096).

    Args:
        out_dim: Output dimension
        text_num_embeds: Number of text embeddings (vocabulary size)
        mask_padding: Whether to mask filler and batch padding tokens
    """

    def __init__(
        self,
        out_dim: int,
        text_num_embeds: int,
        mask_padding: bool = True,
    ):
        super().__init__()
        # Will use 0 as filler token
        self.text_embed = nn.Embedding(text_num_embeds + 1, out_dim)
        self.mask_padding = mask_padding

        self.precompute_max_pos = 1024
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(out_dim, self.precompute_max_pos),
            persistent=False,
        )

    def forward(
        self,
        text: torch.Tensor,  # int["b nt"]
        drop_text: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for MMDiT text embedding.

        Args:
            text: Input text token indices [batch, text_len]
            drop_text: Whether to drop text (for classifier-free guidance)

        Returns:
            Embedded text tensor [batch, text_len, out_dim]
        """
        # Use 0 as filler token. Preprocess of batch pad -1, see list_str_to_idx()
        text = text + 1

        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b nt -> b nt d

        # Sinus pos emb
        batch_start = torch.zeros((text.shape[0],), dtype=torch.long, device=text.device)
        batch_text_len = text.shape[1]
        pos_idx = get_pos_embed_indices(batch_start, batch_text_len, max_pos=self.precompute_max_pos)
        text_pos_embed = self.freqs_cis[pos_idx]

        text = text + text_pos_embed

        if self.mask_padding:
            text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)

        return text


class InputEmbedding(nn.Module):
    """
    Noised input audio and context mixing embedding.

    Used by DiT and UNetT backbones. Concatenates noised input,
    condition audio, and text embedding, then projects to output dimension.

    Args:
        mel_dim: Dimension of mel spectrogram features
        text_dim: Dimension of text embeddings
        out_dim: Output dimension
    """

    def __init__(self, mel_dim: int, text_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: torch.Tensor,  # float["b n d"] - noised input
        cond: torch.Tensor,  # float["b n d"] - condition audio
        text_embed: torch.Tensor,  # float["b n d"] - text embedding
        drop_audio_cond: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for input embedding.

        Args:
            x: Noised input audio [batch, seq_len, mel_dim]
            cond: Condition audio [batch, seq_len, mel_dim]
            text_embed: Text embedding [batch, seq_len, text_dim]
            drop_audio_cond: Whether to drop audio condition (for cfg)

        Returns:
            Combined embedding [batch, seq_len, out_dim]
        """
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class AudioEmbedding(nn.Module):
    """
    Audio embedding for MMDiT backbone.

    Processes noised input and masked condition audio.
    Simpler than InputEmbedding as it doesn't include text.

    Args:
        in_dim: Input dimension (mel_dim)
        out_dim: Output dimension
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(2 * in_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def forward(
        self,
        x: torch.Tensor,  # float["b n d"] - noised input
        cond: torch.Tensor,  # float["b n d"] - condition audio
        drop_audio_cond: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for audio embedding.

        Args:
            x: Noised input audio [batch, seq_len, mel_dim]
            cond: Condition audio [batch, seq_len, mel_dim]
            drop_audio_cond: Whether to drop audio condition (for cfg)

        Returns:
            Combined embedding [batch, seq_len, out_dim]
        """
        if drop_audio_cond:
            cond = torch.zeros_like(cond)

        x = torch.cat((x, cond), dim=-1)
        x = self.linear(x)
        x = self.conv_pos_embed(x) + x
        return x
