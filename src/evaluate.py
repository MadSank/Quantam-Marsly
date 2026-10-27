"""Evaluation pipeline computing physical-unit metrics, paired CIs, summary tables, and plots."""

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

from src.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Formats DataFrame as standard GitHub markdown table without requiring tabulate."""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_str = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                row_str.append(f"{val:.4f}")
            else:
                row_str.append(str(val))
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)


def compute_run_metrics(
    pred_df: pd.DataFrame,
    target_cols: list[str],
    safe_slope_thresh: float = 8.0,
    safe_hazard_thresh: float = 0.05,
) -> dict[str, float]:
    """Computes regression and classification metrics in physical units for a single run."""
    metrics: dict[str, float] = {}

    for t in target_cols:
        y_true = pred_df[f"true_{t}"].to_numpy(dtype=np.float64)
        y_pred = pred_df[f"pred_{t}"].to_numpy(dtype=np.float64)

        if len(y_true) > 0 and not np.isnan(y_true).all():
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
        else:
            rmse, mae, r2 = float("nan"), float("nan"), float("nan")

        metrics[f"RMSE_{t}"] = rmse
        metrics[f"MAE_{t}"] = mae
        metrics[f"R2_{t}"] = r2

    # Landing safety classification metrics
    true_safe = pred_df["true_safe"].to_numpy(dtype=int)
    pred_safe = pred_df["pred_safe"].to_numpy(dtype=int)

    n_truly_safe = int(np.sum(true_safe == 1))
    if n_truly_safe > 0:
        safe_recall = float(np.sum((true_safe == 1) & (pred_safe == 1)) / n_truly_safe)
    else:
        safe_recall = float("nan")

    n_pred_safe = int(np.sum(pred_safe == 1))
    if n_pred_safe > 0:
        false_safe_rate = float(np.sum((pred_safe == 1) & (true_safe == 0)) / n_pred_safe)
    else:
        false_safe_rate = float("nan")

    metrics["safe_recall"] = safe_recall
    metrics["false_safe_rate"] = false_safe_rate

    return metrics


def compute_paired_comparison(
    runs_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric_cols: list[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Computes paired differences, Student-t 95% CIs, and bootstrap CIs across matched runs."""
    df_a = runs_df[runs_df["model_id"] == model_a].copy()
    df_b = runs_df[runs_df["model_id"] == model_b].copy()

    if len(df_a) == 0 or len(df_b) == 0:
        return pd.DataFrame()

    merged = pd.merge(df_a, df_b, on=["fold", "seed"], suffixes=("_a", "_b"))
    if len(merged) == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    for metric in metric_cols:
        col_a = f"{metric}_a"
        col_b = f"{metric}_b"
        if col_a not in merged or col_b not in merged:
            continue

        diffs = merged[col_a].to_numpy(dtype=np.float64) - merged[col_b].to_numpy(dtype=np.float64)
        n = len(diffs)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

        if n >= 2 and std_diff > 1e-12:
            t_crit = float(stats.t.ppf(0.975, df=n - 1))
            se = std_diff / np.sqrt(n)
            ci_t_low = mean_diff - t_crit * se
            ci_t_high = mean_diff + t_crit * se

            boot_means = [
                float(np.mean(rng.choice(diffs, size=n, replace=True))) for _ in range(n_boot)
            ]
            ci_boot_low = float(np.percentile(boot_means, 2.5))
            ci_boot_high = float(np.percentile(boot_means, 97.5))

            t_stat, p_val = stats.ttest_rel(merged[col_a], merged[col_b])
            ci_excludes_zero = bool((ci_t_low > 0) or (ci_t_high < 0))
        else:
            ci_t_low, ci_t_high = float("nan"), float("nan")
            ci_boot_low, ci_boot_high = float("nan"), float("nan")
            p_val = float("nan")
            ci_excludes_zero = False

        records.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "n_pairs": n,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "ci_t_low": ci_t_low,
                "ci_t_high": ci_t_high,
                "ci_boot_low": ci_boot_low,
                "ci_boot_high": ci_boot_high,
                "p_value": float(p_val) if not np.isnan(p_val) else float("nan"),
                "ci_excludes_zero": ci_excludes_zero,
            }
        )

    return pd.DataFrame(records)


def generate_evaluation_plots(
    all_preds_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Generates publication-quality evaluation figures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Predicted vs Measured Scatter Plot
    if len(all_preds_df) > 0 and "true_mean_slope_5m" in all_preds_df.columns:
        plt.figure(figsize=(8, 7))
        models = all_preds_df["model_id"].unique()
        for m in models:
            sub = all_preds_df[all_preds_df["model_id"] == m]
            plt.scatter(
                sub["true_mean_slope_5m"],
                sub["pred_mean_slope_5m"],
                alpha=0.4,
                label=m,
                s=20,
            )

        min_val = min(
            float(all_preds_df["true_mean_slope_5m"].min()),
            float(all_preds_df["pred_mean_slope_5m"].min()),
        )
        max_val = max(
            float(all_preds_df["true_mean_slope_5m"].max()),
            float(all_preds_df["pred_mean_slope_5m"].max()),
        )
        plt.plot([min_val, max_val], [min_val, max_val], "k--", label="Ideal (y=x)", lw=1.5)

        plt.xlabel("Measured 5m Horn Slope (deg)")
        plt.ylabel("Predicted 5m Horn Slope (deg)")
        plt.title("Predicted vs Measured Mars Terrain Slope")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(out_dir / "predicted_vs_measured.png", dpi=200)
        plt.close()

    # 2. Metric Bars Plot
    if len(summary_df) > 0:
        plt.figure(figsize=(10, 6))
        models = summary_df["model_id"].tolist()
        x = np.arange(len(models))

        metrics_to_plot = [
            ("RMSE_mean_slope_5m_mean", "Slope RMSE (deg)"),
            ("safe_recall_mean", "Safe Recall"),
            ("false_safe_rate_mean", "False Safe Rate"),
        ]

        width = 0.25
        for i, (col, lbl) in enumerate(metrics_to_plot):
            if col in summary_df.columns:
                vals = summary_df[col].to_numpy()
                err_col = col.replace("_mean", "_std")
                errs = summary_df[err_col].to_numpy() if err_col in summary_df.columns else None
                plt.bar(x + i * width, vals, width, yerr=errs, capsize=4, label=lbl)

        plt.xticks(x + width, models, rotation=35, ha="right")
        plt.ylabel("Metric Value")
        plt.title("Model Performance Summary with Across-Run Standard Deviation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "metric_bars.png", dpi=200)
        plt.close()

    # 3. Paired Differences Forest Plot
    if len(paired_df) > 0 and "ci_t_low" in paired_df.columns:
        valid_paired = paired_df.dropna(subset=["ci_t_low", "ci_t_high"])
        if len(valid_paired) > 0:
            plt.figure(figsize=(9, 6))
            y_pos = np.arange(len(valid_paired))

            plt.axvline(0, color="gray", linestyle="--", lw=1)
            for idx, (_, row) in enumerate(valid_paired.iterrows()):
                label = f"{row['model_a']} vs {row['model_b']} ({row['metric']})"
                mean_d = row["mean_diff"]
                low = row["ci_t_low"]
                high = row["ci_t_high"]
                color = "crimson" if row["ci_excludes_zero"] else "royalblue"

                plt.errorbar(
                    mean_d,
                    idx,
                    xerr=[[mean_d - low], [high - mean_d]],
                    fmt="o",
                    color=color,
                    capsize=5,
                    label=label if idx == 0 else "",
                )

            plt.yticks(
                y_pos,
                [
                    f"{r['model_a'].replace('model', 'M')} vs {r['model_b'].split('_')[-1]} [{r['metric'][:8]}]"
                    for _, r in valid_paired.iterrows()
                ],
            )
            plt.xlabel("Paired Difference (Model A - Model B) with 95% CI")
            plt.title("Paired Difference Confidence Intervals")
            plt.tight_layout()
            plt.savefig(out_dir / "paired_differences_ci.png", dpi=200)
            plt.close()


def evaluate_all(config: dict[str, Any], smoke: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discovers all test prediction files, computes metrics and CIs, and exports tables and figures."""
    runs_dir = Path(config["paths"]["runs_dir"])
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    pred_files = list(runs_dir.glob("**/test_predictions.parquet"))
    if not pred_files:
        logging.warning("No test_predictions.parquet files found in %s", runs_dir)
        empty_summary = pd.DataFrame()
        empty_paired = pd.DataFrame()
        return empty_summary, empty_paired

    logging.info("Found %d run prediction files in %s", len(pred_files), runs_dir)

    all_preds_list: list[pd.DataFrame] = []
    run_records: list[dict[str, Any]] = []

    target_cols = config["preprocessing"]["target_cols"]
    safe_slope = float(config["preprocessing"]["safe_mean_slope_deg"])
    safe_hazard = float(config["preprocessing"]["safe_hazard_fraction"])

    for p in pred_files:
        df = pd.read_parquet(p)
        model_name = str(df["model_name"].iloc[0])
        variant = str(df["variant"].iloc[0])
        fold = int(df["fold"].iloc[0])
        seed = int(df["seed"].iloc[0])
        model_id = f"{model_name}_{variant}"

        df["model_id"] = model_id
        all_preds_list.append(df)

        metrics = compute_run_metrics(
            df,
            target_cols=target_cols,
            safe_slope_thresh=safe_slope,
            safe_hazard_thresh=safe_hazard,
        )
        run_records.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "variant": variant,
                "fold": fold,
                "seed": seed,
                **metrics,
            }
        )

    all_preds_df = pd.concat(all_preds_list, ignore_index=True)
    runs_df = pd.DataFrame(run_records)

    metric_cols = [c for c in runs_df.columns if c not in {"model_id", "model_name", "variant", "fold", "seed"}]
    summary_rows: list[dict[str, Any]] = []

    for model_id, grp in runs_df.groupby("model_id"):
        row = {"model_id": model_id, "n_runs": len(grp)}
        for col in metric_cols:
            row[f"{col}_mean"] = float(grp[col].mean())
            row[f"{col}_std"] = float(grp[col].std(ddof=1)) if len(grp) > 1 else 0.0
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(by="model_id").reset_index(drop=True)
    summary_csv = results_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logging.info("Saved metrics summary to %s", summary_csv)

    # Save readable Markdown table without requiring tabulate
    summary_md = results_dir / "summary_table.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Model Performance Summary\n\n")
        f.write(dataframe_to_markdown(summary_df))
        f.write("\n")

    matched_pairs = [
        ("model2_qcnn", "model2_linear_control"),
        ("model2_qcnn", "model2_mlp_control"),
        ("model2_qcnn", "model2_no_entanglement_control"),
        ("model2_qcnn", "model2_permuted_qubits_control"),
        ("model3_quanv", "model3_classical_filter_control"),
        ("model3_quanv", "model3_plain_cnn_control"),
        ("model4_pure_qcnn", "model4_tiny_cnn_control"),
        ("model4_pure_qcnn", "model4_mlp_control"),
    ]

    paired_dfs: list[pd.DataFrame] = []
    for m_a, m_b in matched_pairs:
        p_df = compute_paired_comparison(runs_df, m_a, m_b, metric_cols=metric_cols)
        if len(p_df) > 0:
            paired_dfs.append(p_df)

    if paired_dfs:
        all_paired_df = pd.concat(paired_dfs, ignore_index=True)
    else:
        all_paired_df = pd.DataFrame(
            columns=[
                "model_a",
                "model_b",
                "metric",
                "n_pairs",
                "mean_diff",
                "std_diff",
                "ci_t_low",
                "ci_t_high",
                "ci_boot_low",
                "ci_boot_high",
                "p_value",
                "ci_excludes_zero",
            ]
        )

    paired_csv = results_dir / "paired_comparison.csv"
    all_paired_df.to_csv(paired_csv, index=False)
    logging.info("Saved paired comparisons to %s", paired_csv)

    generate_evaluation_plots(all_preds_df, summary_df, all_paired_df, results_dir)

    return summary_df, all_paired_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mars terrain models and compute paired CIs.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--smoke", action="store_true", help="Evaluate smoke run outputs.")
    parser.add_argument("--override", nargs="*", default=None, help="Config overrides key=value.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override, smoke=args.smoke)
    evaluate_all(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
