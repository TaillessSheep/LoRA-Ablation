#!/usr/bin/env python
import os
import json
import argparse
from typing import Dict, Any, List

# ---------------------------------------------------------------------
# 1. Define the expected experiments (based on our earlier design)
# ---------------------------------------------------------------------
EXPECTED_EXPERIMENTS = [
    # Baseline
    "baseline_full_ft",

    # Rank ablation
    "lora_rank_r2",
    "lora_rank_r4",
    "lora_rank_r8",
    "lora_rank_r16",

    # Target modules ablation
    "lora_target_QV",
    "lora_target_QV_fc1_improved",

    # Alpha ablation
    "lora_alpha_8",
    "lora_alpha_16",
    "lora_alpha_32",

    # Strict ablation A/B (you should have these once added to your run script)
    "lora_dropout_only",        # baseline + B
    "lora_improved_dropout",    # A + B (Ours)
]


def safe_load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read JSON {path}: {e}")
        return {}


def collect_experiment(exp_dir: str, exp_name: str) -> Dict[str, Any]:
    """
    Collects metrics + hyperparams from one experiment directory.
    Expects at least results.json. hyperparams.json is optional.
    """
    results_path = os.path.join(exp_dir, "results.json")
    hyper_path = os.path.join(exp_dir, "hyperparams.json")

    results = safe_load_json(results_path)
    hyper = safe_load_json(hyper_path)

    if not results:
        # no usable results, skip
        return {}

    row: Dict[str, Any] = {}
    row["exp_name"] = exp_name

    # Metrics: try several keys safely
    row["eval_accuracy"] = results.get("eval_accuracy", results.get("accuracy", None))
    row["eval_loss"] = results.get("eval_loss", None)

    # Parameters
    row["total_params"] = results.get("total_params", hyper.get("total_params", None))
    row["trainable_params"] = results.get("trainable_params", hyper.get("trainable_params", None))
    row["trainable_ratio"] = results.get("trainable_ratio", hyper.get("trainable_ratio", None))

    # Time / memory
    row["training_time_sec"] = results.get("training_time_sec", None)
    row["gpu_mem_before_mb"] = results.get("gpu_mem_before_mb", None)
    row["gpu_mem_after_mb"] = results.get("gpu_mem_after_mb", None)
    row["gpu_mem_peak_mb"] = results.get("gpu_mem_peak_mb", None)

    # LoRA-related hyperparams (from hyperparams.json, if any)
    # These might be None for full fine-tuning baseline.
    row["use_lora"] = hyper.get("use_lora", None)
    row["rank"] = hyper.get("rank", None)
    row["alpha"] = hyper.get("alpha", None)
    row["dropout"] = hyper.get("dropout", None)
    row["improved"] = hyper.get("improved", None)

    # Training hyperparams
    row["batch"] = hyper.get("batch", None)
    row["lr"] = hyper.get("lr", None)
    row["epochs"] = hyper.get("epochs", None)

    return row


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        print("[WARN] No rows to write to CSV.")
        return
    # determine columns as union of all keys
    cols = sorted({k for r in rows for k in r.keys()})
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v) if v is not None else "")
            f.write(",".join(vals) + "\n")
    print(f"[INFO] CSV summary written to: {path}")


def write_markdown(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        print("[WARN] No rows to write to Markdown.")
        return

    # Focus on a subset of columns for readability
    preferred_cols = [
        "exp_name",
        "eval_accuracy",
        "eval_loss",
        "trainable_params",
        "trainable_ratio",
        "training_time_sec",
        "gpu_mem_peak_mb",
        "use_lora",
        "rank",
        "alpha",
        "dropout",
        "improved",
    ]

    # Only keep columns that actually exist in at least one row
    cols = [c for c in preferred_cols if any(c in r for r in rows)]
    if not cols:
        cols = sorted({k for r in rows for k in r.keys()})

    lines = []
    # header
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")

    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v) if v is not None else "")
        lines.append("| " + " | ".join(vals) + " |")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Markdown summary written to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Collect LoRA experiment results and check missing runs.")
    parser.add_argument(
        "--root",
        type=str,
        default="./outputs",
        help="Root directory containing experiment subfolders (default: ./outputs)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results_summary.csv",
        help="Output CSV file name (default: results_summary.csv)",
    )
    parser.add_argument(
        "--md",
        type=str,
        default="results_summary.md",
        help="Output Markdown file name (default: results_summary.md)",
    )
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"[ERROR] Root directory not found: {root}")
        return

    rows: List[Dict[str, Any]] = []
    found_experiments = set()

    # Scan subdirectories in root
    for name in sorted(os.listdir(root)):
        exp_dir = os.path.join(root, name)
        if not os.path.isdir(exp_dir):
            continue
        results_path = os.path.join(exp_dir, "results.json")
        if not os.path.isfile(results_path):
            # skip folders without results
            continue

        row = collect_experiment(exp_dir, name)
        if not row:
            continue

        rows.append(row)
        found_experiments.add(name)

    # Write outputs
    if rows:
        write_csv(rows, args.csv)
        write_markdown(rows, args.md)
    else:
        print("[WARN] No experiments with results.json found. Nothing to summarize.")
        return

    # Check missing / extra experiments
    expected_set = set(EXPECTED_EXPERIMENTS)
    missing = sorted(expected_set - found_experiments)
    extra = sorted(found_experiments - expected_set)

    print("\n========== Experiment Coverage Check ==========")
    print(f"Expected experiments (from script): {len(expected_set)}")
    print(f"Found experiments with results.json: {len(found_experiments)}")

    if missing:
        print("\n[WARNING] Missing expected experiments:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("\n[OK] No expected experiments are missing.")

    if extra:
        print("\n[INFO] Additional experiments found (not in EXPECTED_EXPERIMENTS):")
        for e in extra:
            print(f"  - {e}")
    else:
        print("\n[OK] No extra experiments found beyond the expected set.")

    print("===============================================\n")


if __name__ == "__main__":
    main()
