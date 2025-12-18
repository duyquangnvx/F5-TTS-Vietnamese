"""
Training process management for F5-TTS fine-tuning.

This module handles starting, stopping, and monitoring training processes.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from importlib.resources import files
from typing import Generator, Optional, Tuple

import gradio as gr
import psutil
import torch

from f5_tts.train.finetune_gradio.utils import path_data, path_project_ckpts, save_settings


# Module-level state
training_process: Optional[subprocess.Popen] = None
stop_signal: bool = False
system = platform.system()
python_executable = sys.executable or "python"
file_train = str(files("f5_tts").joinpath("train/finetune_cli.py"))


def terminate_process_tree(pid: int, including_parent: bool = True) -> None:
    """Terminate a process and all its children."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)
        except OSError:
            pass


def terminate_process(pid: int) -> None:
    """Terminate a process based on the operating system."""
    if system == "Windows":
        cmd = f"taskkill /t /f /pid {pid}"
        os.system(cmd)
    else:
        terminate_process_tree(pid)


def start_training(
    dataset_name: str = "",
    exp_name: str = "F5TTS_v1_Base",
    learning_rate: float = 1e-5,
    batch_size_per_gpu: int = 1,
    batch_size_type: str = "sample",
    max_samples: int = 64,
    grad_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    epochs: int = 100,
    num_warmup_updates: int = 100,
    save_per_updates: int = 500,
    keep_last_n_checkpoints: int = -1,
    last_per_updates: int = 100,
    finetune: bool = True,
    file_checkpoint_train: str = "",
    tokenizer_type: str = "pinyin",
    tokenizer_file: str = "",
    mixed_precision: str = "fp16",
    stream: bool = False,
    logger: str = "wandb",
    ch_8bit_adam: bool = False,
) -> Generator[Tuple[str, gr.update, gr.update], None, None]:
    """Start the training process."""
    global training_process, stop_signal

    # Import here to avoid circular imports
    from f5_tts.api import F5TTS

    # Clear any existing TTS API to free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    path_project = os.path.join(path_data, dataset_name)

    if not os.path.isdir(path_project):
        yield (
            f"There is not project with name {dataset_name}",
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
        return

    file_raw = os.path.join(path_project, "raw.arrow")
    if not os.path.isfile(file_raw):
        yield f"There is no file {file_raw}", gr.update(interactive=True), gr.update(interactive=False)
        return

    # Check if a training process is already running
    if training_process is not None:
        return "Train run already!", gr.update(interactive=False), gr.update(interactive=True)

    yield "start train", gr.update(interactive=False), gr.update(interactive=False)

    # Determine tokenizer type from dataset name if not specified
    if tokenizer_file == "":
        if dataset_name.endswith("_pinyin"):
            tokenizer_type = "pinyin"
        elif dataset_name.endswith("_char"):
            tokenizer_type = "char"
    else:
        tokenizer_type = "custom"

    dataset_name = dataset_name.replace("_pinyin", "").replace("_char", "")

    # Build the training command
    fp16 = f"--mixed_precision={mixed_precision}" if mixed_precision != "none" else ""

    cmd = (
        f"accelerate launch {fp16} {file_train} --exp_name {exp_name}"
        f" --learning_rate {learning_rate}"
        f" --batch_size_per_gpu {batch_size_per_gpu}"
        f" --batch_size_type {batch_size_type}"
        f" --max_samples {max_samples}"
        f" --grad_accumulation_steps {grad_accumulation_steps}"
        f" --max_grad_norm {max_grad_norm}"
        f" --epochs {epochs}"
        f" --num_warmup_updates {num_warmup_updates}"
        f" --save_per_updates {save_per_updates}"
        f" --keep_last_n_checkpoints {keep_last_n_checkpoints}"
        f" --last_per_updates {last_per_updates}"
        f" --dataset_name {dataset_name}"
    )

    if finetune:
        cmd += " --finetune"

    if file_checkpoint_train != "":
        cmd += f" --pretrain {file_checkpoint_train}"

    if tokenizer_file != "":
        cmd += f" --tokenizer_path {tokenizer_file}"

    cmd += f" --tokenizer {tokenizer_type}"
    cmd += f" --log_samples --logger {logger}"

    if ch_8bit_adam:
        cmd += " --bnb_optimizer"

    print("run command : \n" + cmd + "\n")

    save_settings(
        dataset_name,
        exp_name,
        learning_rate,
        batch_size_per_gpu,
        batch_size_type,
        max_samples,
        grad_accumulation_steps,
        max_grad_norm,
        epochs,
        num_warmup_updates,
        save_per_updates,
        keep_last_n_checkpoints,
        last_per_updates,
        finetune,
        file_checkpoint_train,
        tokenizer_type,
        tokenizer_file,
        mixed_precision,
        logger,
        ch_8bit_adam,
    )

    try:
        if not stream:
            # Start the training process
            training_process = subprocess.Popen(cmd, shell=True)

            time.sleep(5)
            yield "train start", gr.update(interactive=False), gr.update(interactive=True)

            # Wait for the training process to finish
            training_process.wait()
        else:
            yield from _stream_training_output(cmd)

        time.sleep(1)

        if training_process is None:
            text_info = "train stop"
        else:
            text_info = "train complete !"

    except Exception as e:
        text_info = f"An error occurred: {str(e)}"

    training_process = None

    yield text_info, gr.update(interactive=True), gr.update(interactive=False)


def _stream_training_output(cmd: str) -> Generator[Tuple[str, gr.update, gr.update], None, None]:
    """Stream training output in real-time."""
    global training_process, stop_signal

    def stream_output(pipe, output_queue):
        try:
            for line in iter(pipe.readline, ""):
                output_queue.put(line)
        except Exception as e:
            output_queue.put(f"Error reading pipe: {str(e)}")
        finally:
            pipe.close()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    training_process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env
    )
    yield "Training started...", gr.update(interactive=False), gr.update(interactive=True)

    stdout_queue = queue.Queue()
    stderr_queue = queue.Queue()

    stdout_thread = threading.Thread(target=stream_output, args=(training_process.stdout, stdout_queue))
    stderr_thread = threading.Thread(target=stream_output, args=(training_process.stderr, stderr_queue))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    stop_signal = False

    while True:
        if stop_signal:
            training_process.terminate()
            time.sleep(0.5)
            if training_process.poll() is None:
                training_process.kill()
            yield "Training stopped by user.", gr.update(interactive=True), gr.update(interactive=False)
            break

        process_status = training_process.poll()

        # Handle stdout
        try:
            while True:
                output = stdout_queue.get_nowait()
                print(output, end="")
                match = re.search(
                    r"Epoch (\d+)/(\d+):\s+(\d+)%\|.*\[(\d+:\d+)<.*?loss=(\d+\.\d+), update=(\d+)", output
                )
                if match:
                    current_epoch = match.group(1)
                    total_epochs = match.group(2)
                    percent_complete = match.group(3)
                    elapsed_time = match.group(4)
                    loss = match.group(5)
                    current_update = match.group(6)
                    message = (
                        f"Epoch: {current_epoch}/{total_epochs}, "
                        f"Progress: {percent_complete}%, "
                        f"Elapsed Time: {elapsed_time}, "
                        f"Loss: {loss}, "
                        f"Update: {current_update}"
                    )
                    yield message, gr.update(interactive=False), gr.update(interactive=True)
                elif output.strip():
                    yield output, gr.update(interactive=False), gr.update(interactive=True)
        except queue.Empty:
            pass

        # Handle stderr
        try:
            while True:
                error_output = stderr_queue.get_nowait()
                print(error_output, end="")
                if error_output.strip():
                    yield f"{error_output.strip()}", gr.update(interactive=False), gr.update(interactive=True)
        except queue.Empty:
            pass

        if process_status is not None and stdout_queue.empty() and stderr_queue.empty():
            if process_status != 0:
                yield (
                    f"Process crashed with exit code {process_status}!",
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                )
            else:
                yield "Training complete!", gr.update(interactive=False), gr.update(interactive=True)
            break

        # Small sleep to prevent CPU thrashing
        time.sleep(0.1)

    # Clean up
    training_process.stdout.close()
    training_process.stderr.close()
    training_process.wait()


def stop_training() -> Tuple[str, gr.update, gr.update]:
    """Stop the currently running training process."""
    global training_process, stop_signal

    if training_process is None:
        return "Train not run !", gr.update(interactive=True), gr.update(interactive=False)
    terminate_process_tree(training_process.pid)
    stop_signal = True
    return "train stop", gr.update(interactive=True), gr.update(interactive=False)


def calculate_train(
    name_project: str,
    epochs: int,
    learning_rate: float,
    batch_size_per_gpu: int,
    batch_size_type: str,
    max_samples: int,
    num_warmup_updates: int,
    finetune: bool,
) -> Tuple[int, float, int, int, int, int]:
    """Calculate optimal training parameters based on dataset."""
    path_project = os.path.join(path_data, name_project)
    file_duration = os.path.join(path_project, "duration.json")

    hop_length = 256
    sampling_rate = 24000

    if not os.path.isfile(file_duration):
        return (
            epochs,
            learning_rate,
            batch_size_per_gpu,
            max_samples,
            num_warmup_updates,
            "project not found !",
        )

    with open(file_duration, "r") as file:
        data = json.load(file)

    duration_list = data["duration"]
    max_sample_length = max(duration_list) * sampling_rate / hop_length
    total_samples = len(duration_list)
    total_duration = sum(duration_list)

    # Detect GPU capabilities
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        total_memory = 0
        for i in range(gpu_count):
            gpu_properties = torch.cuda.get_device_properties(i)
            total_memory += gpu_properties.total_memory / (1024**3)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        gpu_count = torch.xpu.device_count()
        total_memory = 0
        for i in range(gpu_count):
            gpu_properties = torch.xpu.get_device_properties(i)
            total_memory += gpu_properties.total_memory / (1024**3)
    elif torch.backends.mps.is_available():
        gpu_count = 1
        total_memory = psutil.virtual_memory().available / (1024**3)
    else:
        gpu_count = 1
        total_memory = 8  # Assume 8GB as default

    avg_gpu_memory = total_memory / gpu_count

    # Rough estimate of batch size
    if batch_size_type == "frame":
        batch_size_per_gpu = max(int(38400 * (avg_gpu_memory - 5) / 75), int(max_sample_length))
    elif batch_size_type == "sample":
        batch_size_per_gpu = int(200 / (total_duration / total_samples))

    if total_samples < 64:
        max_samples = int(total_samples * 0.25)

    num_warmup_updates = max(num_warmup_updates, int(total_samples * 0.05))

    # Take 1.2M updates as the maximum
    max_updates = 1200000

    if batch_size_type == "frame":
        mini_batch_duration = batch_size_per_gpu * gpu_count * hop_length / sampling_rate
        updates_per_epoch = total_duration / mini_batch_duration
    elif batch_size_type == "sample":
        updates_per_epoch = total_samples / batch_size_per_gpu / gpu_count
    else:
        updates_per_epoch = total_samples

    epochs = int(max_updates / updates_per_epoch)

    if finetune:
        learning_rate = 1e-5
    else:
        learning_rate = 7.5e-5

    return (
        epochs,
        learning_rate,
        batch_size_per_gpu,
        max_samples,
        num_warmup_updates,
        total_samples,
    )


def is_training_running() -> bool:
    """Check if a training process is currently running."""
    return training_process is not None
