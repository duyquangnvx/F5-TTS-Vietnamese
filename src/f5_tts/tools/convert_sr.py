"""
Convert audio sample rate to 24kHz.

This tool converts audio files to 24kHz sample rate for F5-TTS training.
"""

from __future__ import annotations

import glob
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path
from shutil import move
from typing import List, Optional

import click
from tqdm import tqdm

from f5_tts.constants import TARGET_SAMPLE_RATE


def convert_single_file(audio_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> Optional[str]:
    """
    Convert a single audio file to target sample rate.

    Args:
        audio_path: Path to the audio file.
        target_sr: Target sample rate (default: 24000).

    Returns:
        Path to the converted file, or None if conversion failed.
    """
    audio_path = Path(audio_path)
    output_path = audio_path.with_name(f"{audio_path.stem}_{target_sr // 1000}k.wav")

    try:
        subprocess.run(
            ["sox", str(audio_path), "-r", str(target_sr), "-c", "1", str(output_path)],
            check=True,
            capture_output=True,
        )
        return str(output_path)
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        raise RuntimeError("sox is not installed. Please install sox: apt-get install sox libsox-fmt-all")


def remove_original(audio_path: str, suffix: str = "_24k") -> None:
    """
    Remove original file if it's not the converted version.

    Args:
        audio_path: Path to the audio file.
        suffix: Suffix of converted files to preserve.
    """
    if f"{suffix}.wav" not in audio_path:
        os.remove(audio_path)


def rename_converted(audio_path: str, suffix: str = "_24k") -> str:
    """
    Remove the conversion suffix from filename.

    Args:
        audio_path: Path to the audio file.
        suffix: Suffix to remove.

    Returns:
        New path after renaming.
    """
    audio_path = Path(audio_path)
    new_path = audio_path.with_name(audio_path.stem.replace(suffix, "") + ".wav")
    move(str(audio_path), str(new_path))
    return str(new_path)


def process_files_parallel(
    function,
    file_paths: List[str],
    num_workers: int = 16,
    desc: str = "Processing",
) -> List:
    """
    Process files in parallel using multiprocessing.

    Args:
        function: Function to apply to each file.
        file_paths: List of file paths to process.
        num_workers: Number of worker processes.
        desc: Description for progress bar.

    Returns:
        List of results from the function.
    """
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(function, file_paths), total=len(file_paths), desc=desc))
    return results


@click.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input directory containing audio files",
)
@click.option(
    "--pattern",
    "-p",
    default="*.wav",
    help="Glob pattern for audio files (default: *.wav)",
)
@click.option(
    "--sample-rate",
    "-r",
    default=TARGET_SAMPLE_RATE,
    type=int,
    help=f"Target sample rate (default: {TARGET_SAMPLE_RATE})",
)
@click.option(
    "--workers",
    "-w",
    default=16,
    type=int,
    help="Number of worker processes (default: 16)",
)
@click.option(
    "--keep-original",
    is_flag=True,
    default=False,
    help="Keep original files after conversion",
)
@click.option(
    "--in-place",
    is_flag=True,
    default=False,
    help="Replace original files with converted versions",
)
def main(
    input_dir: str,
    pattern: str,
    sample_rate: int,
    workers: int,
    keep_original: bool,
    in_place: bool,
) -> None:
    """Convert audio files to target sample rate for F5-TTS training."""
    # Find all audio files
    search_pattern = os.path.join(input_dir, pattern)
    wav_paths = glob.glob(search_pattern)

    if not wav_paths:
        click.echo(f"No files found matching pattern: {search_pattern}")
        return

    click.echo(f"Found {len(wav_paths)} audio files")

    # Convert sample rate
    suffix = f"_{sample_rate // 1000}k"

    click.echo(f"Converting to {sample_rate}Hz...")

    def convert_wrapper(path):
        return convert_single_file(path, sample_rate)

    process_files_parallel(convert_wrapper, wav_paths, workers, "Converting")

    if in_place:
        # Remove originals and rename converted files
        click.echo("Removing original files...")
        process_files_parallel(
            lambda p: remove_original(p, suffix),
            wav_paths,
            workers,
            "Removing originals",
        )

        # Get converted files
        converted_pattern = os.path.join(input_dir, f"*{suffix}.wav")
        converted_paths = glob.glob(converted_pattern)

        click.echo("Renaming converted files...")
        process_files_parallel(
            lambda p: rename_converted(p, suffix),
            converted_paths,
            workers,
            "Renaming",
        )
    elif not keep_original:
        click.echo("Removing original files...")
        process_files_parallel(
            lambda p: remove_original(p, suffix),
            wav_paths,
            workers,
            "Removing originals",
        )

    click.echo("Done!")


if __name__ == "__main__":
    main()
