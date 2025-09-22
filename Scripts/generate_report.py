#!/usr/bin/env python
# Scripts/generate_report.py
import pandas as pd
from jinja2 import Template
import argparse
import datetime
import os
import base64
import shutil
import json

MODEL_DISPLAY = ["LOGISTIC", "RF", "XGB", "SVM"]

# Per-model plots (kept)
PLOTS = ["confusion_matrix", "feature_importance", "roc_pr_curve", "shap_summary", "gridsearch_panels"]

# Global multi-panel / overview assets we may emit from the pipeline
GLOBAL_ASSETS = {
    "GRID_OVERVIEW": "gridsearch_overview.png",   # all models, multi-panel (already produced)
    "CALIBRATION_GRID": "calibration_grid.png",   # 2x2 (new)
    "PCA_TSNE": "pca_tsne.png",                   # PCA+t-SNE (optional; trainer emits when called)
    # Future optional rollups you might create:
    "CM_GRID": "confusion_grid.png",
    "ROCPR_GRID": "rocpr_grid.png",
    "FI_GRID": "feature_importance_grid.png",
    "SHAP_GRID": "shap_grid.png",
}

def _b64(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_images_dict(figs_dir: str):
    images = {}
    if not figs_dir:
        return images

    # Per-model images
    for plot in PLOTS:
        for model in MODEL_DISPLAY:
            key = f"{plot}_{model}"
            path = os.path.join(figs_dir, f"{plot.lower()}_{model.lower()}.png")
            images[key] = _b64(path)

    # Global/overview images
    for key, fname in GLOBAL_ASSETS.items():
        images[key] = _b64(os.path.join(figs_dir, fname))

    return images

def format_df(df_in: pd.DataFrame):
    """Normalize columns, coerce types, and sort by log loss (ascending)."""
    df = df_in.copy()

    # Required fields (new schema)
    needed = ["model","auc_roc","accuracy","precision","recall","log_loss"]
    lower_cols = [c.lower() for c in df.columns]
    missing = [k for k in needed if k not in lower_cols]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}\nHave: {list(df.columns)}")

    # unify names (accept mixed-casing inputs)
    rename_map = {}
    for k in ["model","auc_roc","accuracy","precision","recall","log_loss","best_cv_logloss","best_params"]:
        if k in lower_cols:
            src = [c for c in df.columns if c.lower() == k][0]
            rename_map[src] = k
    df = df.rename(columns=rename_map)

    # coerce numeric
    for k in ["auc_roc","accuracy","precision","recall","log_loss","best_cv_logloss"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")

    # normalize model labels
    df["model"] = df["model"].astype(str).str.upper()
    order = pd.CategoricalDtype(MODEL_DISPLAY, ordered=True)
    df["model"] = df["model"].astype(order)

    # sort by primary metric: log loss (lower better), tie-break by AUC
    df = df.sort_values(["log_loss","auc_roc"], ascending=[True, False]).reset_index(drop=True)

    # pretty display copy
    df_disp = df.rename(columns={
        "model": "Model",
        "auc_roc": "AUC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "log_loss": "LogLoss",
        "best_cv_logloss": "BestCV_LogLoss",
        "best_params": "BestParams",
    })
    return df, df_disp

def copy_assets_for_portable_report(template_path: str, output_path: str):
    tpl_dir = os.path.dirname(os.path.abspath(template_path))
    out_dir = os.path.dirname(os.path.abspath(output_path))
    for fname in ("style.css", "functionality.js"):
        src = os.path.join(tpl_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))

def load_eda_summary(out_dir):
    eda_json = os.path.join(out_dir, "eda_summary.json")
    if not os.path.exists(eda_json):
        return None
    with open(eda_json, "r") as f:
        return json.load(f)

def add_eda_images(images, figs_dir, eda):
    if not eda or not figs_dir:
        return images
    for k in ["eda_target_dist", "eda_corr", "eda_missing", "eda_feature_hists", "eda_feature_hists_by_class"]:
        fname = eda.get("figs", {}).get(k)
        images[k] = _b64(os.path.join(figs_dir, fname)) if fname else None
    return images

def preview_csv(path, n=10):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df.head(n).to_dict(orient="records"), list(df.columns)
    except Exception:
        return None

def find_grid_csvs(figs_dir: str):
    out = {}
    if not figs_dir:
        return out
    for m in MODEL_DISPLAY:
        p = os.path.join(figs_dir, f"gridsearch_results_{m.lower()}.csv")
        out[m] = os.path.basename(p) if os.path.exists(p) else None
    return out

def generate_html_report(csv_path, output_path, template_path, figs_dir=None):
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    df_raw = pd.read_csv(csv_path)
    df, df_disp = format_df(df_raw)

    eda = load_eda_summary(out_dir)
    images = build_images_dict(figs_dir) if figs_dir else {}
    images = add_eda_images(images, figs_dir, eda)
    grid_csvs = find_grid_csvs(figs_dir) if figs_dir else {}

    # best (first row after sort)
    top = df_disp.iloc[0].to_dict()
    summary = {
        "best_model": top["Model"],
        "best_logloss": float(top["LogLoss"]),
        "best_auc": float(top["AUC"]),
        "best_precision": float(top["Precision"]),
        "best_recall": float(top["Recall"]),
        "date": str(datetime.datetime.now().date()),
        "n_models": int(df_disp.shape[0]),
    }

    # EDA cards + small CSV previews
    eda_cards = None
    t1_preview = None
    gt_preview = None
    if eda:
        eda_cards = {
            "n_samples": eda.get("n_samples"),
            "n_features": eda.get("n_features"),
            "imbalance_ratio": eda.get("imbalance_ratio"),
            "entropy_bits": eda.get("target_entropy_bits"),
            "class_counts": eda.get("class_counts", {}),
            "class_pct": eda.get("class_pct", {}),
            "table1_csv": eda.get("artifacts", {}).get("table1_csv"),
            "group_tests_csv": eda.get("artifacts", {}).get("group_tests_csv"),
        }
        t1_path = os.path.join(out_dir, eda_cards["table1_csv"]) if eda_cards["table1_csv"] else None
        gt_path = os.path.join(out_dir, eda_cards["group_tests_csv"]) if eda_cards["group_tests_csv"] else None
        if t1_path:
            p = preview_csv(t1_path, n=12)
            if p:
                rows, cols = p
                t1_preview = {"rows": rows, "cols": cols, "path": os.path.basename(t1_path)}
        if gt_path:
            p = preview_csv(gt_path, n=12)
            if p:
                rows, cols = p
                gt_preview = {"rows": rows, "cols": cols, "path": os.path.basename(gt_path)}

    with open(template_path, "r", encoding="utf-8") as f:
        template = Template(f.read())

    rendered = template.render(
        date=summary["date"],
        rows=df_disp.to_dict(orient="records"),
        images=images,
        summary=summary,
        eda=eda_cards,
        t1_preview=t1_preview,
        gt_preview=gt_preview,
        grid_csvs=grid_csvs,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)

    copy_assets_for_portable_report(template_path, output_path)
    print(f"📄 HTML report generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from model comparison results.")
    parser.add_argument("--csv", type=str, required=True, help="Path to model comparison CSV file.")
    parser.add_argument("--output", type=str, default="output/report.html", help="Output HTML report file path.")
    parser.add_argument("--template", type=str, default="Scripts/Frontend/index.html", help="Path to Jinja2 HTML template.")
    parser.add_argument("--figs_dir", type=str, default="output/figs", help="Directory containing figure PNGs.")
    args = parser.parse_args()
    generate_html_report(args.csv, args.output, args.template, args.figs_dir)

if __name__ == "__main__":
    main()
