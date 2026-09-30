# GenRevisit

Training code for SFT and RL with NACA.

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
