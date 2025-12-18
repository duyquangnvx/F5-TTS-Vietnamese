"""
Prepare metadata for training datasets.

This tool creates metadata.csv and vocab files from audio datasets.
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import Optional, Set, Tuple

import click
import soundfile as sf
from tqdm import tqdm


def get_audio_duration(wav_path: str) -> float:
    """
    Calculate the duration of an audio file.

    Args:
        wav_path: Path to the WAV file.

    Returns:
        Duration in seconds.
    """
    audio_data, sr = sf.read(wav_path)
    return len(audio_data) / sr


def process_dataset(
    dataset_dir: str,
    training_dir: str,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    min_words: int = 3,
) -> Tuple[int, int]:
    """
    Process dataset: copy WAVs, create metadata and vocab.

    Args:
        dataset_dir: Source directory containing WAV and TXT files.
        training_dir: Destination directory for training data.
        min_duration: Minimum audio duration in seconds.
        max_duration: Maximum audio duration in seconds.
        min_words: Minimum number of words in transcript.

    Returns:
        Tuple of (processed_count, skipped_count).
    """
    wavs_dir = os.path.join(training_dir, "wavs")
    metadata_path = os.path.join(training_dir, "metadata.csv")
    vocab_path = os.path.join(training_dir, "vocab.txt")

    os.makedirs(wavs_dir, exist_ok=True)

    wav_paths = glob.glob(os.path.join(dataset_dir, "*.wav"))
    tokens: Set[str] = set()

    processed_count = 0
    skipped_count = 0

    with open(metadata_path, "w", encoding="utf-8") as fw:
        for wav_path in tqdm(wav_paths, desc="Processing dataset"):
            wav_name = os.path.basename(wav_path)
            wav_dest_path = os.path.join(wavs_dir, wav_name)

            # Read transcript
            txt_path = wav_path.replace(".wav", ".txt")
            if not os.path.exists(txt_path):
                skipped_count += 1
                continue

            with open(txt_path, "r", encoding="utf-8") as fr:
                text = fr.readline().strip().lower()
                text = text.replace("_", " ")
                text = " ".join(text.split())

            # Validate duration and text length
            try:
                duration = get_audio_duration(wav_path)
            except Exception:
                skipped_count += 1
                continue

            if duration < min_duration or duration > max_duration:
                skipped_count += 1
                continue

            if len(text.split()) < min_words:
                skipped_count += 1
                continue

            # Copy audio file
            shutil.copy(wav_path, wav_dest_path)

            # Write to metadata
            fw.write(f"wavs/{wav_name}|{text}\n")

            # Collect tokens for vocab
            tokens.update(text)
            processed_count += 1

    # Write vocab
    with open(vocab_path, "w", encoding="utf-8") as fw_vocab:
        fw_vocab.write("\n".join(sorted(tokens)))

    return processed_count, skipped_count


@click.command()
@click.option(
    "--dataset-dir",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Source directory containing WAV and TXT files",
)
@click.option(
    "--training-dir",
    "-t",
    required=True,
    type=click.Path(),
    help="Destination directory for training data",
)
@click.option(
    "--min-duration",
    default=1.0,
    type=float,
    help="Minimum audio duration in seconds (default: 1.0)",
)
@click.option(
    "--max-duration",
    default=30.0,
    type=float,
    help="Maximum audio duration in seconds (default: 30.0)",
)
@click.option(
    "--min-words",
    default=3,
    type=int,
    help="Minimum number of words in transcript (default: 3)",
)
def main(
    dataset_dir: str,
    training_dir: str,
    min_duration: float,
    max_duration: float,
    min_words: int,
) -> None:
    """Prepare metadata for F5-TTS training datasets."""
    click.echo(f"Processing dataset from: {dataset_dir}")
    click.echo(f"Output directory: {training_dir}")

    processed, skipped = process_dataset(
        dataset_dir=dataset_dir,
        training_dir=training_dir,
        min_duration=min_duration,
        max_duration=max_duration,
        min_words=min_words,
    )

    metadata_path = os.path.join(training_dir, "metadata.csv")
    vocab_path = os.path.join(training_dir, "vocab.txt")

    click.echo(f"\nProcessed: {processed} files")
    click.echo(f"Skipped: {skipped} files")
    click.echo(f"Metadata saved to: {metadata_path}")
    click.echo(f"Vocab saved to: {vocab_path}")


if __name__ == "__main__":
    main()
