"""
Main Gradio application for F5-TTS fine-tuning.

This module assembles all UI components and creates the Gradio interface.
"""

from __future__ import annotations

import click
import gradio as gr

from f5_tts.train.finetune_gradio.utils import (
    get_list_projects,
    create_data_project,
    load_settings,
    check_user,
    check_finetune,
    get_audio_select,
)
from f5_tts.train.finetune_gradio.training_manager import (
    start_training,
    stop_training,
    calculate_train,
)
from f5_tts.train.finetune_gradio.callbacks import (
    transcribe_all,
    create_metadata,
    vocab_check,
    vocab_count,
    vocab_extend,
    extract_and_save_ema_model,
    get_random_sample_prepare,
    get_random_sample_transcribe,
    get_random_sample_infer,
    infer,
    get_checkpoints_project,
    get_audio_project,
    get_combined_stats,
)


def create_app() -> gr.Blocks:
    """Create and return the Gradio application."""
    with gr.Blocks() as app:
        gr.Markdown(
            """
# F5 TTS Automatic Finetune

This is a local web UI for F5 TTS finetuning support. This app supports the following TTS models:

* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)
* [E2 TTS](https://arxiv.org/abs/2406.18009) (Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS)

The pretrained checkpoints support English and Chinese.

For tutorial and updates check here (https://github.com/SWivid/F5-TTS/discussions/143)
"""
        )

        with gr.Row():
            projects, projects_select = get_list_projects()
            tokenizer_type = gr.Radio(label="Tokenizer Type", choices=["pinyin", "char", "custom"], value="pinyin")
            project_name = gr.Textbox(label="Project Name", value="my_speak")
            bt_create = gr.Button("Create a New Project")

        with gr.Row():
            cm_project = gr.Dropdown(
                choices=projects, value=projects_select, label="Project", allow_custom_value=True, scale=6
            )
            ch_refresh_project = gr.Button("Refresh", scale=1)

        bt_create.click(fn=create_data_project, inputs=[project_name, tokenizer_type], outputs=[cm_project])

        with gr.Tabs():
            # Transcribe Data Tab
            with gr.TabItem("Transcribe Data"):
                gr.Markdown("""```plaintext
Skip this step if you have your dataset, metadata.csv, and a folder wavs with all the audio files.
```""")

                ch_manual = gr.Checkbox(label="Audio from Path", value=False)

                mark_info_transcribe = gr.Markdown(
                    """```plaintext
     Place your 'wavs' folder and 'metadata.csv' file in the '{your_project_name}' directory.

     my_speak/
     │
     └── dataset/
         ├── audio1.wav
         └── audio2.wav
         ...
     ```""",
                    visible=False,
                )

                audio_speaker = gr.File(label="Voice", type="filepath", file_count="multiple")
                txt_lang = gr.Text(label="Language", value="English")
                bt_transcribe = gr.Button("Transcribe")
                txt_info_transcribe = gr.Text(label="Info", value="")
                bt_transcribe.click(
                    fn=transcribe_all,
                    inputs=[cm_project, audio_speaker, txt_lang, ch_manual],
                    outputs=[txt_info_transcribe],
                )
                ch_manual.change(fn=check_user, inputs=[ch_manual], outputs=[audio_speaker, mark_info_transcribe])

                random_sample_transcribe = gr.Button("Random Sample")

                with gr.Row():
                    random_text_transcribe = gr.Text(label="Text")
                    random_audio_transcribe = gr.Audio(label="Audio", type="filepath")

                random_sample_transcribe.click(
                    fn=get_random_sample_transcribe,
                    inputs=[cm_project],
                    outputs=[random_text_transcribe, random_audio_transcribe],
                )

            # Vocab Check Tab
            with gr.TabItem("Vocab Check"):
                gr.Markdown("""```plaintext
Check the vocabulary for fine-tuning Emilia_ZH_EN to ensure all symbols are included. For fine-tuning a new language.
```""")

                check_button = gr.Button("Check Vocab")
                txt_info_check = gr.Text(label="Info", value="")

                gr.Markdown("""```plaintext
Using the extended model, you can finetune to a new language that is missing symbols in the vocab. This creates a new model with a new vocabulary size and saves it in your ckpts/project folder.
```""")

                exp_name_extend = gr.Radio(
                    label="Model", choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"], value="F5TTS_v1_Base"
                )

                with gr.Row():
                    txt_extend = gr.Textbox(
                        label="Symbols",
                        value="",
                        placeholder="To add new symbols, make sure to use ',' for each symbol",
                        scale=6,
                    )
                    txt_count_symbol = gr.Textbox(label="New Vocab Size", value="", scale=1)

                extend_button = gr.Button("Extend")
                txt_info_extend = gr.Text(label="Info", value="")

                txt_extend.change(vocab_count, inputs=[txt_extend], outputs=[txt_count_symbol])
                check_button.click(fn=vocab_check, inputs=[cm_project], outputs=[txt_info_check, txt_extend])
                extend_button.click(
                    fn=vocab_extend, inputs=[cm_project, txt_extend, exp_name_extend], outputs=[txt_info_extend]
                )

            # Prepare Data Tab
            with gr.TabItem("Prepare Data"):
                gr.Markdown("""```plaintext
Skip this step if you have your dataset, raw.arrow, duration.json, and vocab.txt
```""")

                gr.Markdown(
                    """```plaintext
     Place all your "wavs" folder and your "metadata.csv" file in your project name directory.

     Supported audio formats: "wav", "mp3", "aac", "flac", "m4a", "alac", "ogg", "aiff", "wma", "amr"

     Example wav format:
     my_speak/
     │
     ├── wavs/
     │   ├── audio1.wav
     │   └── audio2.wav
     |   ...
     │
     └── metadata.csv

     File format metadata.csv:

     audio1|text1 or audio1.wav|text1 or your_path/audio1.wav|text1
     audio2|text1 or audio2.wav|text1 or your_path/audio2.wav|text1
     ...

     ```"""
                )
                ch_tokenizern = gr.Checkbox(label="Create Vocabulary", value=False, visible=False)

                bt_prepare = gr.Button("Prepare")
                txt_info_prepare = gr.Text(label="Info", value="")
                txt_vocab_prepare = gr.Text(label="Vocab", value="")

                bt_prepare.click(
                    fn=create_metadata, inputs=[cm_project, ch_tokenizern], outputs=[txt_info_prepare, txt_vocab_prepare]
                )

                random_sample_prepare = gr.Button("Random Sample")

                with gr.Row():
                    random_text_prepare = gr.Text(label="Tokenizer")
                    random_audio_prepare = gr.Audio(label="Audio", type="filepath")

                random_sample_prepare.click(
                    fn=get_random_sample_prepare, inputs=[cm_project], outputs=[random_text_prepare, random_audio_prepare]
                )

            # Train Model Tab
            with gr.TabItem("Train Model"):
                gr.Markdown("""```plaintext
The auto-setting is still experimental. Set a large value of epoch if not sure; and keep last N checkpoints if limited disk space.
If you encounter a memory error, try reducing the batch size per GPU to a smaller number.
```""")
                with gr.Row():
                    bt_calculate = gr.Button("Auto Settings")
                    lb_samples = gr.Label(label="Samples")
                    batch_size_type = gr.Radio(label="Batch Size Type", choices=["frame", "sample"], value="frame")

                with gr.Row():
                    ch_finetune = gr.Checkbox(label="Finetune", value=True)
                    tokenizer_file = gr.Textbox(label="Tokenizer File", value="")
                    file_checkpoint_train = gr.Textbox(label="Path to the Pretrained Checkpoint", value="")

                with gr.Row():
                    exp_name = gr.Radio(
                        label="Model", choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"], value="F5TTS_v1_Base"
                    )
                    learning_rate = gr.Number(label="Learning Rate", value=1e-5, step=1e-5)

                with gr.Row():
                    batch_size_per_gpu = gr.Number(label="Batch Size per GPU", value=3200)
                    max_samples = gr.Number(label="Max Samples", value=64)

                with gr.Row():
                    grad_accumulation_steps = gr.Number(label="Gradient Accumulation Steps", value=1)
                    max_grad_norm = gr.Number(label="Max Gradient Norm", value=1.0)

                with gr.Row():
                    epochs = gr.Number(label="Epochs", value=100)
                    num_warmup_updates = gr.Number(label="Warmup Updates", value=100)

                with gr.Row():
                    save_per_updates = gr.Number(label="Save per Updates", value=500)
                    keep_last_n_checkpoints = gr.Number(
                        label="Keep Last N Checkpoints",
                        value=-1,
                        step=1,
                        precision=0,
                        info="-1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints",
                    )
                    last_per_updates = gr.Number(label="Last per Updates", value=100)

                with gr.Row():
                    ch_8bit_adam = gr.Checkbox(label="Use 8-bit Adam optimizer")
                    mixed_precision = gr.Radio(label="mixed_precision", choices=["none", "fp16", "bf16"], value="fp16")
                    cd_logger = gr.Radio(label="logger", choices=["wandb", "tensorboard"], value="wandb")
                    start_button = gr.Button("Start Training")
                    stop_button = gr.Button("Stop Training", interactive=False)

                if projects_select is not None:
                    (
                        exp_name_value,
                        learning_rate_value,
                        batch_size_per_gpu_value,
                        batch_size_type_value,
                        max_samples_value,
                        grad_accumulation_steps_value,
                        max_grad_norm_value,
                        epochs_value,
                        num_warmup_updates_value,
                        save_per_updates_value,
                        keep_last_n_checkpoints_value,
                        last_per_updates_value,
                        finetune_value,
                        file_checkpoint_train_value,
                        tokenizer_type_value,
                        tokenizer_file_value,
                        mixed_precision_value,
                        logger_value,
                        bnb_optimizer_value,
                    ) = load_settings(projects_select)

                    # Assigning values to the respective components
                    exp_name.value = exp_name_value
                    learning_rate.value = learning_rate_value
                    batch_size_per_gpu.value = batch_size_per_gpu_value
                    batch_size_type.value = batch_size_type_value
                    max_samples.value = max_samples_value
                    grad_accumulation_steps.value = grad_accumulation_steps_value
                    max_grad_norm.value = max_grad_norm_value
                    epochs.value = epochs_value
                    num_warmup_updates.value = num_warmup_updates_value
                    save_per_updates.value = save_per_updates_value
                    keep_last_n_checkpoints.value = keep_last_n_checkpoints_value
                    last_per_updates.value = last_per_updates_value
                    ch_finetune.value = finetune_value
                    file_checkpoint_train.value = file_checkpoint_train_value
                    tokenizer_type.value = tokenizer_type_value
                    tokenizer_file.value = tokenizer_file_value
                    mixed_precision.value = mixed_precision_value
                    cd_logger.value = logger_value
                    ch_8bit_adam.value = bnb_optimizer_value

                ch_stream = gr.Checkbox(label="Stream Output Experiment", value=True)
                txt_info_train = gr.Text(label="Info", value="")

                list_audios, select_audio = get_audio_project(projects_select, False)

                select_audio_ref = select_audio
                select_audio_gen = select_audio

                if select_audio is not None:
                    select_audio_ref += "_ref.wav"
                    select_audio_gen += "_gen.wav"

                with gr.Row():
                    ch_list_audio = gr.Dropdown(
                        choices=list_audios,
                        value=select_audio,
                        label="Audios",
                        allow_custom_value=True,
                        scale=6,
                        interactive=True,
                    )
                    bt_stream_audio = gr.Button("Refresh", scale=1)
                    bt_stream_audio.click(fn=get_audio_project, inputs=[cm_project], outputs=[ch_list_audio])
                    cm_project.change(fn=get_audio_project, inputs=[cm_project], outputs=[ch_list_audio])

                with gr.Row():
                    audio_ref_stream = gr.Audio(label="Original", type="filepath", value=select_audio_ref)
                    audio_gen_stream = gr.Audio(label="Generate", type="filepath", value=select_audio_gen)

                ch_list_audio.change(
                    fn=get_audio_select,
                    inputs=[ch_list_audio],
                    outputs=[audio_ref_stream, audio_gen_stream],
                )

                start_button.click(
                    fn=start_training,
                    inputs=[
                        cm_project,
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
                        ch_finetune,
                        file_checkpoint_train,
                        tokenizer_type,
                        tokenizer_file,
                        mixed_precision,
                        ch_stream,
                        cd_logger,
                        ch_8bit_adam,
                    ],
                    outputs=[txt_info_train, start_button, stop_button],
                )
                stop_button.click(fn=stop_training, outputs=[txt_info_train, start_button, stop_button])

                bt_calculate.click(
                    fn=calculate_train,
                    inputs=[
                        cm_project,
                        epochs,
                        learning_rate,
                        batch_size_per_gpu,
                        batch_size_type,
                        max_samples,
                        num_warmup_updates,
                        ch_finetune,
                    ],
                    outputs=[
                        epochs,
                        learning_rate,
                        batch_size_per_gpu,
                        max_samples,
                        num_warmup_updates,
                        lb_samples,
                    ],
                )

                ch_finetune.change(
                    check_finetune, inputs=[ch_finetune], outputs=[file_checkpoint_train, tokenizer_file, tokenizer_type]
                )

                def setup_load_settings():
                    output_components = [
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
                        ch_finetune,
                        file_checkpoint_train,
                        tokenizer_type,
                        tokenizer_file,
                        mixed_precision,
                        cd_logger,
                        ch_8bit_adam,
                    ]
                    return output_components

                outputs = setup_load_settings()

                cm_project.change(
                    fn=load_settings,
                    inputs=[cm_project],
                    outputs=outputs,
                )

                ch_refresh_project.click(
                    fn=load_settings,
                    inputs=[cm_project],
                    outputs=outputs,
                )

            # Test Model Tab
            with gr.TabItem("Test Model"):
                gr.Markdown("""```plaintext
SOS: Check the use_ema setting (True or False) for your model to see what works best for you. use seed -1 from random
```""")
                exp_name_test = gr.Radio(
                    label="Model", choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"], value="F5TTS_v1_Base"
                )
                list_checkpoints, checkpoint_select = get_checkpoints_project(projects_select, False)

                with gr.Row():
                    nfe_step = gr.Number(label="NFE Step", value=32)
                    speed = gr.Slider(label="Speed", value=1.0, minimum=0.3, maximum=2.0, step=0.1)
                    seed = gr.Number(label="Seed", value=-1, minimum=-1)
                    remove_silence = gr.Checkbox(label="Remove Silence")

                ch_use_ema = gr.Checkbox(label="Use EMA", value=True)
                with gr.Row():
                    cm_checkpoint = gr.Dropdown(
                        choices=list_checkpoints, value=checkpoint_select, label="Checkpoints", allow_custom_value=True
                    )
                    bt_checkpoint_refresh = gr.Button("Refresh")

                random_sample_infer = gr.Button("Random Sample")

                ref_text = gr.Textbox(label="Ref Text")
                ref_audio = gr.Audio(label="Audio Ref", type="filepath")
                gen_text = gr.Textbox(label="Gen Text")

                random_sample_infer.click(
                    fn=get_random_sample_infer, inputs=[cm_project], outputs=[ref_text, gen_text, ref_audio]
                )

                with gr.Row():
                    txt_info_gpu = gr.Textbox("", label="Device")
                    seed_info = gr.Text(label="Seed :")
                    check_button_infer = gr.Button("Infer")

                gen_audio = gr.Audio(label="Audio Gen", type="filepath")

                check_button_infer.click(
                    fn=infer,
                    inputs=[
                        cm_project,
                        cm_checkpoint,
                        exp_name_test,
                        ref_text,
                        ref_audio,
                        gen_text,
                        nfe_step,
                        ch_use_ema,
                        speed,
                        seed,
                        remove_silence,
                    ],
                    outputs=[gen_audio, txt_info_gpu, seed_info],
                )

                bt_checkpoint_refresh.click(fn=get_checkpoints_project, inputs=[cm_project], outputs=[cm_checkpoint])
                cm_project.change(fn=get_checkpoints_project, inputs=[cm_project], outputs=[cm_checkpoint])

            # Prune Checkpoint Tab
            with gr.TabItem("Prune Checkpoint"):
                gr.Markdown("""```plaintext
Reduce the Base model size from 5GB to 1.3GB. The new checkpoint file prunes out optimizer and etc., can be used for inference or finetuning afterward, but not able to resume pretraining.
```""")
                txt_path_checkpoint = gr.Text(label="Path to Checkpoint:")
                txt_path_checkpoint_small = gr.Text(label="Path to Output:")
                ch_safetensors = gr.Checkbox(label="Safetensors", value="")
                txt_info_reduse = gr.Text(label="Info", value="")
                reduse_button = gr.Button("Reduce")
                reduse_button.click(
                    fn=extract_and_save_ema_model,
                    inputs=[txt_path_checkpoint, txt_path_checkpoint_small, ch_safetensors],
                    outputs=[txt_info_reduse],
                )

            # System Info Tab
            with gr.TabItem("System Info"):
                output_box = gr.Textbox(label="GPU and CPU Information", lines=20)

                def update_stats():
                    return get_combined_stats()

                update_button = gr.Button("Update Stats")
                update_button.click(fn=update_stats, outputs=output_box)

                def auto_update():
                    yield gr.update(value=update_stats())

                gr.update(fn=auto_update, inputs=[], outputs=output_box)

    return app


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
def main(port, host, share, api):
    """Launch the F5-TTS fine-tuning Gradio interface."""
    print("Starting app...")
    app = create_app()
    app.queue(api_open=api).launch(server_name=host, server_port=port, share=share, show_api=api)


if __name__ == "__main__":
    main()
