# PAMPOS-DeepAccident Extension

Research implementation extending PAMPOS's causal-transformer misbehavior
detector with:

- a multi-step "dreaming" (autoregressive rollout) inference mechanism,
- scene-grounded visual input features,
- Chain-of-Thought (CoT) explanation generation,
- environment-level and adversarial-ML attack injection,

evaluated on the DeepAccident benchmark (AAAI-24).

## Status

Implementation in progress. See project plan (kept outside this repo) for
full phase breakdown, source verification, and rationale.

## Environment

- Python 3.12
- Windows / PowerShell
- Local dev/testing: RTX 2050 (4GB VRAM)
- Full-scale training: UCloud (remote)

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt --break-system-packages
```

## Structure

- `data/` — raw and processed DeepAccident data (gitignored)
- `src/model/` — PAMPOS architecture, losses, dreaming mechanism
- `src/data_pipeline/` — dataset loading, scene features, V2X message layer
- `src/attacks/environment_attacks/` — CARLA-native environment/infrastructure attacks
- `src/attacks/adversarial_ml_attacks/` — model/data-targeting adversarial attacks
- `src/cot/` — CoT caption generation and explanation composition
- `src/eval/` — metrics and ablation evaluation
- `scripts/` — entrypoints for download, training, evaluation
- `configs/` — training configuration files
- `outputs/` — checkpoints, logs, results (gitignored)
