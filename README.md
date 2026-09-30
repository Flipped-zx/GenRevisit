# GenRevisit: training code

This repository contains the training components for a multimodal image retry policy. It intentionally ships **code and configuration only**: no trajectories, prompts, images, checkpoints, API credentials, or local machine paths.

## Layout

```text
configs/sft/       Joint SFT recipe and DeepSpeed ZeRO-3 settings
configs/rl/        Pi5v2 NACA/GRPO optimization settings
scripts/           SFT launcher
src/genrevisit/    NACA allocation, action-token objective and optimizer step
tests/             CPU checks for credit and loss behavior
docs/              Source and scope notes
```

The SFT recipe uses the **Joint SFT** training settings (Qwen3-VL-8B-Instruct, full language model SFT, frozen vision/projector, 8 devices, BF16, FA2, ZeRO-3). The RL code uses **Pi5v2 V2 NACA**.

## Joint SFT

Install a compatible LLaMA-Factory environment. The recorded training stack was LLaMA-Factory 0.9.5, Transformers 4.57.6, PyTorch 2.7.1, DeepSpeed 0.18.2, and FlashAttention 2.6.1. Hardware-specific PyTorch builds are environment dependent.

Provide your own prepared LLaMA-Factory dataset (`dataset_info.json`, train/validation files, referenced images) and base model. Training data is not part of this repository.

```bash
python scripts/train_sft.py \
  --model-dir /path/to/Qwen3-VL-8B-Instruct \
  --dataset-dir /path/to/prepared-dataset \
  --output-dir /path/to/output \
  --dry-run

python scripts/train_sft.py \
  --model-dir /path/to/Qwen3-VL-8B-Instruct \
  --dataset-dir /path/to/prepared-dataset \
  --output-dir /path/to/output
```

The launcher substitutes only local paths. Tracking is disabled by default. Run the token/mask audit appropriate to your dataset before training.

## Pi5v2 RL

`genrevisit.train.naca` implements the V2 positive image-action water-filling allocation and negative trajectory advantage broadcast. `genrevisit.train.rl_loss` implements the asymmetric clipped action-token GRPO objective with sampled low-variance reference KL and candidate-level reduction. `genrevisit.train.rl_step` applies one optimizer update to an already prepared batch. The reference configuration is in `configs/rl/pi5v2_naca.yaml`.

The RL interface takes prepared old/reference log probabilities, action masks, group-relative advantages, and image-action quality values.

## Checks

```bash
python -m pytest tests -q
```

The directory split follows the training-oriented layout visible in [Gen-Searcher](https://github.com/tulerfeng/Gen-Searcher) and the package/scripts/configuration layout visible in [GenEvolve](https://github.com/MeiGen-AI/GenEvolve). No code from those repositories is copied here.
