<p align="center">
  <img src="assets/logo.svg" alt="GenRevisit" width="230" />
</p>

<h1 align="center">GenRevisit</h1>

<p align="center">
  Agent interfaces, supervised fine-tuning, and reinforcement learning with NACA.
</p>

<p align="center">
  <a href="https://huggingface.co/GenRevisit/GenRevisit-SFT">🤗 SFT Model</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/GenRevisit/GenRevisit-RL">🤗 RL Model</a>
  <br />
  <a href="https://modelscope.cn/models/GenRevisit/GenRevisit-SFT">ModelScope SFT</a> &nbsp;·&nbsp;
  <a href="https://modelscope.cn/models/GenRevisit/GenRevisit-RL">ModelScope RL</a>
</p>

<p align="center">
  <a href="#intro">Intro</a> &nbsp;·&nbsp;
  <a href="#model">Model</a> &nbsp;·&nbsp;
  <a href="#installation">Installation</a> &nbsp;·&nbsp;
  <a href="#agent">Agent</a> &nbsp;·&nbsp;
  <a href="#sft">SFT</a> &nbsp;·&nbsp;
  <a href="#rl--naca">RL / NACA</a>
</p>

---

## Intro

GenRevisit provides agent field contracts, message rendering, SFT training configuration, and RL training components with NACA.

## Model

| Model | Hugging Face | ModelScope |
| :--- | :---: | :---: |
| **GenRevisit-SFT** | [🤗 Weights](https://huggingface.co/GenRevisit/GenRevisit-SFT) | [Weights](https://modelscope.cn/models/GenRevisit/GenRevisit-SFT) |
| **GenRevisit-RL** | [🤗 Weights](https://huggingface.co/GenRevisit/GenRevisit-RL) | [Weights](https://modelscope.cn/models/GenRevisit/GenRevisit-RL) |

## Installation

```bash
python -m pip install -e .
```

For RL training components:

```bash
python -m pip install -e ".[rl]"
```

## Agent

The agent reads task constraints, candidate images and feedback, then emits one structured action: `query_skill`, `generate_image`, `edit_image`, or `submit_attempt`.

See [Agent](docs/agent.md) for input fields, action schemas, validation and message rendering.

## Structure

```text
GenRevisit/
├── assets/                  Project artwork
├── configs/
│   ├── sft/                 SFT and DeepSpeed configuration
│   └── rl/                  NACA and GRPO configuration
├── docs/                    Agent interface documentation
├── scripts/                 SFT training launcher
├── src/genrevisit/
│   ├── agent/               Field contracts, schemas and message rendering
│   └── train/               Credit allocation, loss and optimizer step
└── tests/                   Agent and training checks
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
python -m pip install -e ".[test]"
python -m pytest tests -q
```
