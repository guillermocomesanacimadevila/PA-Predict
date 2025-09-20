#!/usr/bin/env python
# Scripts/benchmark_models.py

import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, log_loss
)
from pa_model_trainer import PAModelTrainer  # same folder import

plt.switch_backend("Agg")

MODELS = [
    ("logistic", "LOGISTIC"),
    ("rf",       "RF"),
    ("xgb",      "XGB"),
    ("svm",      "SVM"),
]

def ensure_dir(p: str):
    d = os.path.dirname(p) if os.path.splitext(p)[1] else p
    if d:
        os.makedirs(d, exist_ok=True)

def run_one_model(data_path, model_key, model_label, figs_dir, model_out_dir):
    print(f"\n🚀 Benchmarking model: {model_label}")
    trainer = PAModelTrainer(data_path=data_path, model_type=model_key)
    trainer.load_data()
    trainer.train_model()
    trainer.evaluate_model()

    # save model
    model_path = os.path.join(model_out_dir, f"model_{model_key}.pkl")
    ensure_dir(model_path)
    import joblib
    joblib.dump(trainer.model, model_path)

    # standard plots (your trainer already saves grid panels + grid CSV)
    trainer.plot_gridsearch_diagnostics(out_dir=figs_dir)
    trainer.plot_feature_importance(save_path=os.path.join(figs_dir, f"feature_importance_{model_key}.png"))
    trainer.plot_confusion_matrix(save_path=os.path.join(figs_dir, f"confusion_matrix_{model_key}.png"))
    trainer.plot_roc_pr_curves(save_path=os.path.join(figs_dir, f"roc_pr_curve_{model_key}.png"))
    trainer.plot_shap_summary(save_path=os.path.join(figs_dir, f"shap_summary_{model_key}.png"))

    # test metrics
    y_prob = trainer.model.predict_proba(trainer.X_test)[:, 1]
    y_pred = trainer.model.predict(trainer.X_test)

    metrics = dict(
        model=model_label,
        auc_roc=roc_auc_score(trainer.y_test, y_prob),
        accuracy=accuracy_score(trainer.y_test, y_pred),
        precision=precision_score(trainer.y_test, y_pred, zero_division=0),
        recall=recall_score(trainer.y_test, y_pred, zero_division=0),
        log_loss=log_loss(trainer.y_test, y_prob),
        best_params=json.dumps(trainer.grid_obj.best_params_, sort_keys=True),
        best_cv_logloss=(-trainer.grid_obj.best_score_),  # flip neg_log_loss -> positive logloss
    )

    # return paths to grid panel + csv for collage/overview
    grid_panel = os.path.join(figs_dir, f"gridsearch_panels_{model_key}.png")
    grid_csv   = os.path.join(figs_dir, f"gridsearch_results_{model_key}.csv")

    return metrics, grid_panel, grid_csv

def build_grid_overview(panel_paths, out_path):
    """Make a 2x2 collage of per-model grid panels."""
    ensure_dir(out_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for i, p in enumerate(panel_paths):
        ax = axes[i]
        if p and os.path.exists(p):
            img = plt.imread(p)
            ax.imshow(img)
            ax.set_axis_off()
        else:
            ax.text(0.5, 0.5, "Panel not found", ha="center", va="center")
            ax.set_axis_off()
    fig.suptitle("Grid Search Overview (All Models)", y=0.98, fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Benchmark all models + aggregate results.")
    ap.add_argument("--data", required=True, help="CSV with target column 'Diagnosed_PA'.")
    ap.add_argument("--output_csv", default="output/model_comparison.csv", help="Where to write model comparison CSV")
    ap.add_argument("--figs_dir", default="output/figs", help="Where to save figures")
    ap.add_argument("--models_dir", default="output/models", help="Where to save trained models")
    args = ap.parse_args()

    ensure_dir(args.output_csv)
    os.makedirs(args.figs_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    rows = []
    panel_paths = []
    grid_csvs = []

    for key, label in MODELS:
        m, panel, gridcsv = run_one_model(
            data_path=args.data,
            model_key=key,
            model_label=label,
            figs_dir=args.figs_dir,
            model_out_dir=args.models_dir
        )
        rows.append(m)
        panel_paths.append(panel)
        grid_csvs.append(gridcsv)

    # write comparison CSV
    df = pd.DataFrame(rows)
    df = df[["model", "auc_roc", "accuracy", "precision", "recall", "log_loss", "best_cv_logloss", "best_params"]]
    df.sort_values("log_loss", inplace=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\n📊 Model comparison saved to: {args.output_csv}")

    # overview collage
    overview_path = os.path.join(args.figs_dir, "gridsearch_overview.png")
    build_grid_overview(panel_paths, overview_path)
    print(f"🖼️  Grid-search overview saved to: {overview_path}")

if __name__ == "__main__":
    main()
