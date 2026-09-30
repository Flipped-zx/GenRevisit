"""Launch the SFT LLaMA-Factory recipe with local paths."""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-dataset", default="sft_train")
    parser.add_argument("--eval-dataset", default="sft_validation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for label, path in (("model", args.model_dir), ("dataset", args.dataset_dir)):
        if not path.is_dir():
            parser.error(f"{label} directory is missing: {path}")
    config_path = ROOT / "configs/sft/full_sft.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model_name_or_path"] = str(args.model_dir.resolve())
    config["dataset_dir"] = str(args.dataset_dir.resolve())
    config["output_dir"] = str(args.output_dir.resolve())
    config["dataset"] = args.train_dataset
    config["eval_dataset"] = args.eval_dataset
    config["deepspeed"] = str((ROOT / config["deepspeed"]).resolve())
    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty for a fresh training run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(config, tmp, sort_keys=False)
        runtime_path = Path(tmp.name)
    try:
        subprocess.run(["llamafactory-cli", "train", str(runtime_path)], check=True)
    finally:
        runtime_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
