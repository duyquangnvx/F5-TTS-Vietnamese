"""
Check and extend vocabulary for F5-TTS training.

This tool compares dataset vocabulary against pretrained vocabulary
and identifies missing tokens.
"""

from __future__ import annotations

import os
from typing import List, Set, Tuple

import click


def load_vocab(file_path: str) -> List[str]:
    """
    Load vocabulary from a file.

    Args:
        file_path: Path to the vocabulary file.

    Returns:
        List of tokens.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return [line.rstrip("\n") for line in file]


def save_vocab(file_path: str, vocab: List[str]) -> None:
    """
    Save vocabulary to a file.

    Args:
        file_path: Path to the output file.
        vocab: List of tokens to save.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(f"{token}\n" for token in vocab)


def find_missing_tokens(
    pretrained_vocab: List[str],
    dataset_vocab: List[str],
) -> List[str]:
    """
    Find tokens in dataset that are missing from pretrained vocabulary.

    Args:
        pretrained_vocab: List of tokens from pretrained model.
        dataset_vocab: List of tokens from dataset.

    Returns:
        List of missing tokens.
    """
    pretrained_set: Set[str] = set(pretrained_vocab)
    return [token for token in dataset_vocab if token not in pretrained_set]


def merge_vocabularies(
    pretrained_vocab: List[str],
    missing_tokens: List[str],
) -> List[str]:
    """
    Merge pretrained vocabulary with missing tokens.

    Args:
        pretrained_vocab: Original vocabulary.
        missing_tokens: Tokens to add.

    Returns:
        Merged vocabulary list.
    """
    return pretrained_vocab + missing_tokens


@click.command()
@click.option(
    "--pretrained-vocab",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to pretrained vocabulary file",
)
@click.option(
    "--dataset-vocab",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to dataset vocabulary file",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Path to output merged vocabulary file",
)
@click.option(
    "--check-only",
    is_flag=True,
    default=False,
    help="Only check for missing tokens without creating output",
)
def main(
    pretrained_vocab: str,
    dataset_vocab: str,
    output: str,
    check_only: bool,
) -> None:
    """Check vocabulary coverage and optionally extend with missing tokens."""
    click.echo(f"Loading pretrained vocabulary from: {pretrained_vocab}")
    tokens_pretrained = load_vocab(pretrained_vocab)
    click.echo(f"  Pretrained vocab size: {len(tokens_pretrained)}")

    click.echo(f"Loading dataset vocabulary from: {dataset_vocab}")
    tokens_dataset = load_vocab(dataset_vocab)
    click.echo(f"  Dataset vocab size: {len(tokens_dataset)}")

    # Find missing tokens
    missing_tokens = find_missing_tokens(tokens_pretrained, tokens_dataset)

    if not missing_tokens:
        click.echo("\nAll dataset tokens are covered by pretrained vocabulary!")
        click.echo("No vocabulary extension needed.")
        return

    click.echo(f"\nMissing tokens: {len(missing_tokens)}")
    if len(missing_tokens) <= 50:
        click.echo(f"  Tokens: {missing_tokens}")
    else:
        click.echo(f"  First 50 tokens: {missing_tokens[:50]}")
        click.echo(f"  ... and {len(missing_tokens) - 50} more")

    if check_only:
        click.echo("\n(Check only mode - no output file created)")
        return

    # Merge vocabularies
    new_vocab = merge_vocabularies(tokens_pretrained, missing_tokens)

    # Save merged vocabulary
    save_vocab(output, new_vocab)

    click.echo(f"\nNew vocabulary saved to: {output}")
    click.echo(f"  New vocab size: {len(new_vocab)}")
    click.echo(f"  Tokens added: {len(missing_tokens)}")


if __name__ == "__main__":
    main()
