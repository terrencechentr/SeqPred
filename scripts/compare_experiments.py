#!/usr/bin/env python3
"""
Compare experiments in ./experiments/ by reading evaluation/metrics.txt
Generate summary CSV and English-only comparison plots.
"""

import os
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
MODELS_DIR = os.path.join(EXPERIMENTS_DIR, "models")
ANALYSIS_DIR = os.path.join(EXPERIMENTS_DIR, "analysis")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Ensure English fonts (no Chinese)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

# Metrics mapping to keep English keys
KEY_REMAP = {
    "Direction Accuracy (总体)": "Direction Accuracy",
    "Up Direction Accuracy (上涨)": "Up Direction Accuracy",
    "Down Direction Accuracy (下跌)": "Down Direction Accuracy",
    "R² Score": "R2 Score",
}


def parse_metrics_file(path: str) -> Dict[str, float]:
    data: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "===" in line:
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().replace("%", "")
            try:
                num = float(val)
            except ValueError:
                continue
            key = KEY_REMAP.get(key, key)
            data[key] = num
    return data


def load_ar_npz(npz_path: str) -> Optional[dict]:
    try:
        return np.load(npz_path, allow_pickle=True)
    except Exception:
        return None


def ar_metrics_from_npz(npz: dict) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Compute AR metrics on forecast part (after initial_length).
    Prefer price metrics if price arrays exist; else use returns.
    Returns: (MAE, RMSE, MAPE, mode)
    """
    initial_len = int(npz.get("initial_length", 0))

    # pick first interval
    if "pred_prices" in npz and "truth_prices" in npz:
        pred_arr = npz["pred_prices"][0]
        truth_arr = npz["truth_prices"][0]
        mode = "price"
    else:
        pred_arr = npz["predictions_full"][0]
        truth_arr = npz["ground_truth_full"][0]
        mode = "return"

    steps = min(len(pred_arr), len(truth_arr))
    pred = np.array(pred_arr[:steps])
    truth = np.array(truth_arr[:steps])
    if steps <= initial_len:
        return None, None, None, mode

    pred_f = pred[initial_len:]
    truth_f = truth[initial_len:]
    mae = float(np.mean(np.abs(pred_f - truth_f)))
    rmse = float(np.sqrt(np.mean((pred_f - truth_f) ** 2)))
    mape = None
    if mode == "price":
        denom = np.maximum(np.abs(truth_f), 1e-10)
        mape = float(np.mean(np.abs((truth_f - pred_f) / denom)) * 100)
    return mae, rmse, mape, mode


def collect_experiments() -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    scan_dir = MODELS_DIR if os.path.isdir(MODELS_DIR) else EXPERIMENTS_DIR
    for name in sorted(os.listdir(scan_dir)):
        exp_dir = os.path.join(scan_dir, name)
        metrics_path = os.path.join(exp_dir, "evaluation", "metrics.txt")
        if not os.path.isfile(metrics_path):
            continue
        metrics = parse_metrics_file(metrics_path)
        if not metrics:
            continue
        row = {"experiment": name}
        row.update(metrics)

        # autoregressive metrics
        ar_path = os.path.join(exp_dir, "evaluation", "autoregressive_predictions_multi_test.npz")
        npz = load_ar_npz(ar_path)
        if npz is not None:
            ar_mae, ar_rmse, ar_mape, mode = ar_metrics_from_npz(npz)
            if ar_mae is not None:
                row[f"AR_{mode}_MAE"] = ar_mae
                row[f"AR_{mode}_RMSE"] = ar_rmse
                if ar_mape is not None:
                    row[f"AR_{mode}_MAPE"] = ar_mape
        rows.append(row)
    return pd.DataFrame(rows)


def plot_bar(df: pd.DataFrame, metric: str, ascending: bool, out_path: str, title: str, xlabel: str):
    plt.figure(figsize=(10, 6))
    order_df = df.sort_values(metric, ascending=ascending)
    sns.barplot(data=order_df, y="experiment", x=metric, palette="viridis")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12, fontweight="bold")
    plt.ylabel("Experiment", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_path}")


def plot_ar_overlay(df: pd.DataFrame, out_dir: str, scan_dir: str, max_intervals: int = 3):
    """
    Overlay AR predictions across models, keeping initial context + forecast.
    - Draw up to `max_intervals` intervals (separate plots per interval & mode).
    - Prefer price series if available; fall back to returns.
    """
    # Collect series per model and per interval
    collected = {}
    for name in df["experiment"]:
        exp_dir = os.path.join(scan_dir, name)
        ar_path = os.path.join(exp_dir, "evaluation", "autoregressive_predictions_multi_test.npz")
        npz = load_ar_npz(ar_path)
        if npz is None:
            continue
        initial_len = int(npz.get("initial_length", 0))
        n_intervals = len(npz["predictions_full"])
        for idx in range(min(max_intervals, n_intervals)):
            if "pred_prices" in npz and "truth_prices" in npz:
                pred = np.array(npz["pred_prices"][idx])
                truth = np.array(npz["truth_prices"][idx])
                mode = "price"
            else:
                pred = np.array(npz["predictions_full"][idx])
                truth = np.array(npz["ground_truth_full"][idx])
                mode = "return"
            steps = min(len(pred), len(truth))
            pred = pred[:steps]
            truth = truth[:steps]
            if steps <= initial_len:
                continue
            key = (idx, mode)
            collected.setdefault(key, []).append(
                {
                    "label": name,
                    "x": np.arange(steps),
                    "pred": pred,
                    "truth": truth,
                    "initial_len": initial_len,
                }
            )

    if not collected:
        return

    for (idx, mode), series in collected.items():
        plt.figure(figsize=(12, 6))
        # Plot truth once
        truth = series[0]["truth"]
        x = np.arange(len(truth))
        plt.plot(x, truth, label="Ground Truth", linewidth=2, color="black", alpha=0.6)
        init_len = series[0]["initial_len"]
        plt.axvline(x=init_len - 1, color="gray", linestyle="--", alpha=0.6, label="Initial Context End")
        # Models
        for s in series:
            plt.plot(s["x"], s["pred"], linewidth=2, label=f"{s['label']} pred")
        plt.xlabel("Time Steps (context + forecast)", fontsize=11, fontweight="bold")
        plt.ylabel("Price" if mode == "price" else "Normalized Return", fontsize=11, fontweight="bold")
        plt.title(f"AR Comparison Interval {idx+1} ({mode})", fontsize=13, fontweight="bold")
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"AR_overlay_interval{idx+1}_{mode}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {out_path}")


def main():
    df = collect_experiments()
    if df.empty:
        print("No experiments with evaluation/metrics.txt found.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ANALYSIS_DIR, f"comparison_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Save summary
    summary_csv = os.path.join(out_dir, "summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"✓ Summary saved: {summary_csv}")

    # Plots (only if metric exists)
    if "RMSE" in df.columns:
        plot_bar(df, "RMSE", ascending=True, out_path=os.path.join(out_dir, "RMSE.png"),
                 title="RMSE Comparison", xlabel="RMSE")
    if "MAE" in df.columns:
        plot_bar(df, "MAE", ascending=True, out_path=os.path.join(out_dir, "MAE.png"),
                 title="MAE Comparison", xlabel="MAE")
    if "R2 Score" in df.columns:
        plot_bar(df, "R2 Score", ascending=False, out_path=os.path.join(out_dir, "R2_Score.png"),
                 title="R2 Score Comparison", xlabel="R2 Score")
    if "Direction Accuracy" in df.columns:
        plot_bar(df, "Direction Accuracy", ascending=False, out_path=os.path.join(out_dir, "Direction_Accuracy.png"),
                 title="Direction Accuracy Comparison", xlabel="Direction Accuracy (%)")

    # Dump JSON for convenience
    summary_json = os.path.join(out_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    print(f"✓ Summary saved: {summary_json}")

    # AR overlay plots
    scan_dir = MODELS_DIR if os.path.isdir(MODELS_DIR) else EXPERIMENTS_DIR
    plot_ar_overlay(df, out_dir, scan_dir)


if __name__ == "__main__":
    main()

