# F5-TTS Scripts

Windows scripts for F5-TTS training, inference, and evaluation.

## Scripts Overview

| Script | Description |
|--------|-------------|
| `fine_tuning.bat` | Batch script for full fine-tuning pipeline |
| `fine_tuning.ps1` | PowerShell script for fine-tuning (recommended) |
| `infer.bat` | Batch script for inference |
| `infer.ps1` | PowerShell script for inference (recommended) |
| `eval_batch.bat` | Batch evaluation script |

## Quick Start

### Fine-tuning

**Using PowerShell (recommended):**
```powershell
# Run full pipeline from stage 0
.\scripts\fine_tuning.ps1 -Stage 0 -StopStage 5

# Run only fine-tuning (stage 5)
.\scripts\fine_tuning.ps1 -Stage 5 -StopStage 5

# Custom dataset and model
.\scripts\fine_tuning.ps1 -DatasetName "my_dataset" -ExpName "F5TTS_v1_Base"
```

**Using Batch:**
```cmd
:: Edit the script to set STAGE and STOP_STAGE variables
scripts\fine_tuning.bat
```

### Inference

**Using PowerShell:**
```powershell
# Basic inference
.\scripts\infer.ps1 -GenText "Xin chào các bạn"

# Custom parameters
.\scripts\infer.ps1 `
    -RefAudio "my_ref.wav" `
    -RefText "Reference text" `
    -GenText "Text to generate" `
    -Model "F5TTS_v1_Base" `
    -Speed 0.9
```

**Using Batch:**
```cmd
:: Edit infer.bat to set your parameters
scripts\infer.bat
```

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| 0 | Convert audio sample rate to 24kHz |
| 1 | Prepare metadata (metadata.csv, vocab.txt) |
| 2 | Check vocabulary against pretrained model |
| 3 | Extend model embeddings for new vocabulary |
| 4 | Extract features (prepare_csv_wavs) |
| 5 | Run fine-tuning |

## Output Directory

All outputs are saved to the `output/` directory:

```
output/
├── inference/      # Inference outputs
├── eval/           # Evaluation results
└── logs/           # Training logs
```

## Check Device (GPU Detection)

Before running training or inference, you can check which GPU is available:

```powershell
# Check selected device
python -m f5_tts.tools.check_device

# Show all available devices
python -m f5_tts.tools.check_device --all

# Test CUDA is working
python -m f5_tts.tools.check_device --test
```

Example output:
```
==================================================
Device Information
==================================================
  Device Type: CUDA
  Device Name: NVIDIA GeForce RTX 5070
  Memory: 12.00 GB
  Compute Capability: 12.0
  CUDA Version: 12.4
  Multiprocessors: 48
  Dtype: torch.float16
==================================================
```

## Requirements

- Python 3.9+
- CUDA (for GPU training)
- Sox (for audio conversion, install via chocolatey: `choco install sox`)

## Notes

1. **PowerShell Execution Policy**: If you get an execution policy error, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **GPU Selection**: Set `CUDA_VISIBLE_DEVICES` environment variable to select GPU:
   ```powershell
   $env:CUDA_VISIBLE_DEVICES = "0"  # Use first GPU
   ```

3. **Multi-GPU Training**: Uncomment the `accelerate launch` section in the scripts.
