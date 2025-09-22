#!/usr/bin/env python
# Scripts/benchmark_models.py

import argparse, os, json, math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, log_loss
)

from pa_model_trainer import PAModelTrainer  # same-folder import

# Headless-safe + clean aesthetics
plt.switch_backend("Agg")
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "semibold",
    "axes.labelsize": 11, "axes.linewidth": 1.0,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 10
})

# Model list: (key used by trainer, label used in figures/report)
MODELS = [
    ("logistic", "LOGISTIC"),
    ("rf",       "RF"),
    ("xgb",      "XGB"),
    ("svm",      "SVM"),
]

MODEL_COLORS = {
    "LOGISTIC": "#4464ad",
    "SVM":      "#e07a5f",
    "RF":       "#3d9970",
    "XGB":      "#9d4edd",
}

PARAM_LABEL = {
    "param_C": "C",
    "param_gamma": "gamma",
    "param_n_estimators": "n_estimators",
    "param_max_depth": "max_depth",
    "param_learning_rate": "learning_rate",
}

# ─────────────────────────── utils ─────────────────────────── #
def ensure_dir(p: str):
    d = os.path.dirname(p) if os.path.splitext(p)[1] else p
    if d:
        os.makedirs(d, exist_ok=True)

def _is_numeric_list(vals):
    try:
        x = pd.to_numeric(pd.Series(list(vals)), errors="coerce")
        return x.notna().all()
    except Exception:
        return False

def _ci95(std, n):
    std = np.asarray(std, dtype=float)
    n = np.asarray(n, dtype=float)
    return 1.96 * (std / np.sqrt(np.maximum(1.0, n)))

# ─────────── grid-search overview (one figure; all models) ─────────── #
def make_gridsearch_overview(cv_tables: dict, out_path: str):
    """
    cv_tables: {"LOGISTIC": df, "RF": df, "XGB": df, "SVM": df}
    Each df = pd.DataFrame(GridSearchCV.cv_results_) with:
      - param_* columns
      - mean_test_score (neg_log_loss)
      - std_test_score
      - optionally mean_loss (+) and std_folds
    Saves a single multi-panel figure with one subplot per hyperparameter
    (with >= 2 unique values across any model), each subplot showing a line
    per model with 95% CI ribbons.
    """
    if not cv_tables:
        return

    # Normalize frames to have mean_loss (+) and std_folds
    dfs = {}
    for name, df in cv_tables.items():
        t = df.copy()
        if "mean_loss" not in t:
            t["mean_loss"] = (-t["mean_test_score"]).astype(float)
        if "std_folds" not in t and "std_test_score" in t:
            t["std_folds"] = t["std_test_score"]
        dfs[name] = t

    # Union of hyperparameters with at least 2 unique values
    union_vals = {}
    for t in dfs.values():
        for c in [c for c in t.columns if c.startswith("param_")]:
            union_vals.setdefault(c, set()).update(pd.unique(t[c]))
    union_vals = {p: v for p, v in union_vals.items() if len(v) >= 2}
    if not union_vals:
        return  # nothing to plot

    # Build x-grids
    xgrid, xticklabels = {}, {}
    for p, vals in union_vals.items():
        if _is_numeric_list(vals):
            xs = np.sort(pd.to_numeric(list(vals)))
            xgrid[p] = xs
            xticklabels[p] = None
        else:
            labels = [str(v) for v in vals]
            xgrid[p] = np.arange(len(labels))
            xticklabels[p] = labels

    # Series per param, per model (fallback baseline when model lacks that param)
    series_by_param = {p: {} for p in xgrid.keys()}
    for model_name, t in dfs.items():
        baseline_y = t["mean_loss"].min()
        baseline_ci = _ci95(t["mean_loss"].std(ddof=1) if t.shape[0] > 1 else 0.0, max(1, t.shape[0]))
        model_params = [c for c in t.columns if c.startswith("param_") and c in xgrid]

        for p in xgrid.keys():
            xs = xgrid[p]
            labels = xticklabels[p]
            if p in model_params:
                grp = t.groupby(p, dropna=False)["mean_loss"].agg(["min", "std", "count"]).reset_index()
                grp.rename(columns={"min": "y"}, inplace=True)

                if labels is None:
                    key_vals = pd.to_numeric(grp[p])
                    key_vals = key_vals.to_numpy() if hasattr(key_vals, "to_numpy") else np.asarray(key_vals)
                    order = np.argsort(key_vals)
                    grp = grp.iloc[order]
                    y_map  = dict(zip(key_vals, grp["y"].to_numpy()))
                    ci_arr = _ci95(grp["std"].fillna(0.0), grp["count"].clip(lower=1))
                    ci_map = dict(zip(key_vals, ci_arr))
                    y  = np.array([y_map.get(v, np.nan) for v in xs])
                    ci = np.array([ci_map.get(v, np.nan) for v in xs])
                else:
                    keys   = grp[p].astype(str).tolist()
                    y_map  = dict(zip(keys, grp["y"].to_numpy()))
                    ci_arr = _ci95(grp["std"].fillna(0.0), grp["count"].clip(lower=1))
                    ci_map = dict(zip(keys, ci_arr))
                    y  = np.array([y_map.get(lbl, np.nan) for lbl in labels])
                    ci = np.array([ci_map.get(lbl, np.nan) for lbl in labels])
            else:
                # model lacks this hyperparam → flat baseline
                y  = np.full_like(xs, baseline_y, dtype=float)
                ci = np.full_like(xs, float(baseline_ci), dtype=float)

            series_by_param[p][model_name] = {"x": xs, "y": y, "ci": ci, "labels": labels}

    # Order panels
    params_order = ["param_C", "param_gamma", "param_n_estimators", "param_max_depth", "param_learning_rate"]
    params_order = [p for p in params_order if p in xgrid] + [p for p in xgrid if p not in params_order]

    # Plot
    n = len(params_order)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 4.9 * nrows), squeeze=False)

    model_order = ["LOGISTIC", "SVM", "RF", "XGB"]

    for i, p in enumerate(params_order):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ymin, ymax = np.inf, -np.inf

        for m in model_order:
            if m not in series_by_param[p]:
                continue
            s = series_by_param[p][m]
            xs, ys, ci, labels = s["x"], s["y"], s["ci"], s["labels"]
            color = MODEL_COLORS.get(m, None)

            ax.plot(xs, ys, marker="o", lw=2.0, label=m, color=color)
            ax.fill_between(xs, ys - ci, ys + ci, alpha=0.18, color=color)

            ymin = min(ymin, float(np.nanmin(ys - ci)))
            ymax = max(ymax, float(np.nanmax(ys + ci)))

            if labels is not None:
                ax.set_xticks(xs)
                ax.set_xticklabels(labels)

        pad = 0.06 * (ymax - ymin + 1e-12)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(PARAM_LABEL.get(p, p.replace("param_", "")))
        ax.set_xlabel("Value")
        if c == 0:
            ax.set_ylabel("Log-loss ↓")
        ax.spines["left"].set_alpha(0.7)
        ax.spines["bottom"].set_alpha(0.7)

    # Hide empties
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    # Global legend
    handles, labels = [], []
    for ax in axes.ravel():
        if not handles:
            handles, labels = ax.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    ensure_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"🖼️  Grid-search overview (all models) saved to: {out_path}")

# ───────────────────────── model run ───────────────────────── #
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

    # per-model assets (trainer will also save its own per-model grid panels/csv)
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
        best_cv_logloss=(-trainer.grid_obj.best_score_),  # neg_log_loss -> positive
    )

    # return cv_results for unified overview + probs for calibration
    cv_df    = trainer.cv_results_df.copy()
    y_true   = trainer.y_test.copy()
    y_probs  = y_prob.copy()

    # Optional: panel/csv paths (already generated by the trainer)
    grid_panel = os.path.join(figs_dir, f"gridsearch_panels_{model_key}.png")
    grid_csv   = os.path.join(figs_dir, f"gridsearch_results_{model_key}.csv")

    return metrics, cv_df, (y_true, y_probs), grid_panel, grid_csv

# ─────────────────────────── main ─────────────────────────── #
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
    cv_tables = {}          # for unified grid-search overview
    probas_dict = {}        # for 2x2 calibration grid
    per_model_panels = []   # kept if you later want to collage per-model panels
    grid_csvs = []

    for key, label in MODELS:
        m, cv_df, (y_true, y_prob), panel_path, grid_csv = run_one_model(
            data_path=args.data,
            model_key=key,
            model_label=label,
            figs_dir=args.figs_dir,
            model_out_dir=args.models_dir
        )
        rows.append(m)
        cv_tables[label] = cv_df
        probas_dict[label] = (y_true, y_prob)
        per_model_panels.append(panel_path)
        grid_csvs.append(grid_csv)

    # write comparison CSV
    df = pd.DataFrame(rows)
    df = df[["model", "auc_roc", "accuracy", "precision", "recall", "log_loss", "best_cv_logloss", "best_params"]]
    df.sort_values("log_loss", inplace=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\n📊 Model comparison saved to: {args.output_csv}")

    # unified grid-search overview (used by HTML as GRID_OVERVIEW)
    overview_path = os.path.join(args.figs_dir, "gridsearch_overview.png")
    make_gridsearch_overview(cv_tables, overview_path)

    # 2x2 calibration grid (one panel per classifier)
    calib_path = os.path.join(args.figs_dir, "calibration_grid.png")
    PAModelTrainer.plot_calibration_grid(probas_dict, calib_path, n_bins=20, loess_frac=0.25)
    print(f"🩺 Calibration grid saved to: {calib_path}")

if __name__ == "__main__":
    main()
