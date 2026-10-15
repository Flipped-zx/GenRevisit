<p align="center">
  <img src="assets/banner.svg" alt="GenRevisit" width="900" />
</p>

<h1 align="center">GenRevisit</h1>

<p align="center">
  Training code and model weights for SFT and RL with NACA.
</p>

<p align="center">
  <a href="https://huggingface.co/GenRevisit/GenRevisit-SFT"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GenRevisit--SFT-0891b2?style=for-the-badge" alt="Hugging Face: GenRevisit-SFT" /></a>
  &nbsp;
  <a href="https://huggingface.co/GenRevisit/GenRevisit-RL"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GenRevisit--RL-7c3aed?style=for-the-badge" alt="Hugging Face: GenRevisit-RL" /></a>
</p>

<p align="center">
  <a href="#model">Model</a> &nbsp;·&nbsp;
  <a href="#structure">Structure</a> &nbsp;·&nbsp;
  <a href="#sft">SFT</a> &nbsp;·&nbsp;
  <a href="#rl--naca">RL / NACA</a> &nbsp;·&nbsp;
  <a href="#tests">Tests</a>
</p>

---

## Model

| Model | Weights |
| :--- | :--- |
| **GenRevisit-SFT** | [🤗 Hugging Face](https://huggingface.co/GenRevisit/GenRevisit-SFT) |
| **GenRevisit-RL** | [🤗 Hugging Face](https://huggingface.co/GenRevisit/GenRevisit-RL) |

## Structure

```text
configs/sft/       SFT and DeepSpeed configuration
configs/rl/        NACA and GRPO configuration
scripts/           SFT training launcher
src/genrevisit/    Credit allocation, loss, and optimizer step
tests/             Small training checks
```

## SFT

Prepare a LLaMA-Factory compatible dataset and a local base model. Then run:

```bash
python scripts/train_sft.py \
  --model-dir "$MODEL_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR"
```

The default dataset names are `sft_train` and `sft_validation`. Use `--train-dataset` and `--eval-dataset` if your `dataset_info.json` uses different names. The launcher reads `configs/sft/full_sft.yaml` and substitutes the local paths.

## RL / NACA

The modules in `src/genrevisit/train/` provide NACA action credit, the GRPO loss, and an optimizer step for prepared training batches. The configuration is in `configs/rl/naca_grpo.yaml`.

## Tests

```bash
PYTHONPATH=src python -m pytest tests -q
```
