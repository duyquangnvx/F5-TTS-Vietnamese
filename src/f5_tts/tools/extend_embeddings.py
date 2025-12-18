"""
Extend model embeddings for new vocabulary tokens.

This tool expands the embedding layer of a pretrained model to accommodate
new tokens, enabling fine-tuning on datasets with expanded vocabulary.
"""

from __future__ import annotations

import os
import random
from typing import List, Optional

import click
import torch
from cached_path import cached_path
from safetensors.torch import load_file

from f5_tts.constants import PRETRAINED_MODELS


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_vocab(file_path: str) -> List[str]:
    """
    Load vocabulary from a file.

    Args:
        file_path: Path to the vocabulary file.

    Returns:
        List of tokens.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines()]


def expand_model_embeddings(
    ckpt_path: str,
    new_ckpt_path: str,
    num_new_tokens: int,
    seed: int = 666,
) -> int:
    """
    Expand model embeddings by adding new token vectors.

    Args:
        ckpt_path: Path to the original checkpoint.
        new_ckpt_path: Path to save the expanded checkpoint.
        num_new_tokens: Number of new tokens to add.
        seed: Random seed for initialization.

    Returns:
        New vocabulary size.
    """
    set_random_seed(seed)

    # Load checkpoint
    if ckpt_path.endswith(".safetensors"):
        ckpt = load_file(ckpt_path, device="cpu")
        ckpt = {"ema_model_state_dict": ckpt}
    elif ckpt_path.endswith(".pt"):
        ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        raise ValueError("Unsupported checkpoint format. Only .safetensors or .pt supported.")

    ema_sd = ckpt.get("ema_model_state_dict", {})
    embed_key = "ema_model.transformer.text_embed.text_embed.weight"

    if embed_key not in ema_sd:
        raise KeyError(f"Embedding key not found in checkpoint: {embed_key}")

    old_embeddings = ema_sd[embed_key]
    vocab_old, embed_dim = old_embeddings.shape
    vocab_new = vocab_old + num_new_tokens

    # Create expanded embeddings
    new_embeddings = torch.zeros((vocab_new, embed_dim))
    new_embeddings[:vocab_old] = old_embeddings
    new_embeddings[vocab_old:] = torch.randn((num_new_tokens, embed_dim))

    ema_sd[embed_key] = new_embeddings

    # Ensure output directory exists
    os.makedirs(os.path.dirname(new_ckpt_path), exist_ok=True)

    # Save checkpoint
    torch.save(ckpt, new_ckpt_path)

    return vocab_new


@click.command()
@click.option(
    "--checkpoint",
    "-c",
    type=str,
    default=None,
    help="Path to checkpoint file, or use --model to download from HuggingFace",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"]),
    default=None,
    help="Pretrained model to download from HuggingFace",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Path to save the expanded checkpoint",
)
@click.option(
    "--pretrained-vocab",
    "-p",
    type=click.Path(exists=True),
    default=None,
    help="Path to pretrained vocabulary file",
)
@click.option(
    "--new-vocab",
    "-n",
    type=click.Path(exists=True),
    default=None,
    help="Path to new (expanded) vocabulary file",
)
@click.option(
    "--num-tokens",
    "-t",
    type=int,
    default=None,
    help="Number of new tokens to add (alternative to vocab files)",
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=666,
    help="Random seed for embedding initialization (default: 666)",
)
def main(
    checkpoint: Optional[str],
    model: Optional[str],
    output: str,
    pretrained_vocab: Optional[str],
    new_vocab: Optional[str],
    num_tokens: Optional[int],
    seed: int,
) -> None:
    """Extend model embeddings to accommodate new vocabulary tokens."""
    # Validate inputs
    if checkpoint is None and model is None:
        raise click.UsageError("Either --checkpoint or --model must be specified")

    if checkpoint is not None and model is not None:
        raise click.UsageError("Only one of --checkpoint or --model can be specified")

    if num_tokens is None and (pretrained_vocab is None or new_vocab is None):
        raise click.UsageError(
            "Either --num-tokens or both --pretrained-vocab and --new-vocab must be specified"
        )

    # Get checkpoint path
    if model is not None:
        click.echo(f"Downloading {model} from HuggingFace...")
        model_url = PRETRAINED_MODELS.get(model)
        if model_url is None:
            raise click.UsageError(f"Unknown model: {model}")
        ckpt_path = str(cached_path(model_url))
    else:
        ckpt_path = checkpoint

    click.echo(f"Using checkpoint: {ckpt_path}")

    # Calculate number of new tokens
    if num_tokens is not None:
        tokens_to_add = num_tokens
    else:
        click.echo(f"Loading pretrained vocabulary from: {pretrained_vocab}")
        tokens_pretrained = load_vocab(pretrained_vocab)
        click.echo(f"  Pretrained vocab size: {len(tokens_pretrained)}")

        click.echo(f"Loading new vocabulary from: {new_vocab}")
        tokens_new = load_vocab(new_vocab)
        click.echo(f"  New vocab size: {len(tokens_new)}")

        tokens_to_add = len(tokens_new) - len(tokens_pretrained)

    if tokens_to_add <= 0:
        click.echo("No new tokens to add. Vocabulary sizes are equal or new vocab is smaller.")
        return

    click.echo(f"\nExpanding embeddings by {tokens_to_add} tokens...")

    new_vocab_size = expand_model_embeddings(
        ckpt_path=ckpt_path,
        new_ckpt_path=output,
        num_new_tokens=tokens_to_add,
        seed=seed,
    )

    click.echo(f"\nCheckpoint saved to: {output}")
    click.echo(f"  Original vocab size: {new_vocab_size - tokens_to_add}")
    click.echo(f"  New vocab size: {new_vocab_size}")
    click.echo(f"  Tokens added: {tokens_to_add}")


if __name__ == "__main__":
    main()
