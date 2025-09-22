#!/usr/bin/env python
# Scripts/pa_model_trainer.py

import argparse
import os
import joblib
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend("Agg")  

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve

# SHAP
try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

# Ibmlearn over-sampling
try:
    from imblearn.over_sampling import RandomOverSampler
    IMB_OK = True
except Exception:
    IMB_OK = False

# LOESS curve for calibration smoothing
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    LOESS_OK = True
except Exception:
    LOESS_OK = False

# ───────────────── Aesthetics ───────────────── #
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.constrained_layout.use": False,   
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "legend.fontsize": 10,
    "axes.prop_cycle": mpl.cycler(color=[
        "#4464ad", "#e07a5f", "#3d9970", "#d38c2c", "#9d4edd", "#2a9d8f"
    ]),
})

# ───────────────── Helpers ───────────────── #
def _ensure_dir(p: str):
    """Create parent directory for a file path or the directory itself if path has no suffix."""
    d = os.path.dirname(p) if os.path.splitext(p)[1] else p
    if d:
        os.makedirs(d, exist_ok=True)

def _to_loss(neg_score):
    """GridSearchCV uses neg_* scorers for losses; flip sign to get positive loss."""
    return -float(neg_score)

def _ci95(std, n):
    std = np.asarray(std, dtype=float)
    n = np.asarray(n, dtype=float)
    return 1.96 * (std / np.sqrt(np.maximum(1.0, n)))

def _entropy_bits(p):
    """Shannon entropy (bits) for binary distribution [p, 1-p]."""
    p = np.clip(float(p), 1e-12, 1 - 1e-12)
    q = 1.0 - p
    return -(p * np.log2(p) + q * np.log2(q))

def _normalized_entropy_bits(p):
    """Entropy normalized to [0,1] (divide by log2(2)=1 for binary)."""
    return _entropy_bits(p) / 1.0

def _expected_calibration_error(y_true, y_prob, n_bins=20):
    """ECE with equal-width bins."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(y_prob, bins, right=True)
    ece = 0.0
    total = len(y_prob)
    for b in range(1, n_bins + 1):
        mask = indices == b
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (np.sum(mask) / total) * abs(acc - conf)
    return ece

def _loess_smooth(x, y, frac=0.25):
    x = np.asarray(x); y = np.asarray(y)
    if LOESS_OK:
        sm = lowess(y, x, frac=frac, it=0, return_sorted=True)
        return sm[:, 0], sm[:, 1]
    # Fallback: simple moving average over sorted x
    order = np.argsort(x)
    x_s = x[order]; y_s = y[order]
    k = max(3, int(len(x) * frac))
    if k % 2 == 0: k += 1
    pad = k // 2
    y_pad = np.pad(y_s, (pad, pad), mode="edge")
    kernel = np.ones(k) / k
    y_ma = np.convolve(y_pad, kernel, mode="valid")
    return x_s, y_ma

# ───────────────── Trainer ───────────────── #
class PAModelTrainer:
    @staticmethod
    def _safe_sample_frame(X, n=200, random_state=42):
        if isinstance(X, pd.DataFrame):
            n = min(len(X), n)
            return X.sample(n=n, random_state=random_state)
        # numpy fallback
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
        return X[idx]

    # ─────────────────────────── Init ─────────────────────────── #
    def __init__(
        self,
        data_path,
        model_type='logistic',
        resample_if_ir_below: float = 40.0,          
        resample_if_norm_entropy_below: float = 0.65 
    ):
        self.data_path = data_path
        self.model_type = model_type
        self.model = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.feature_names = None
        self.grid_obj = None         
        self.cv_results_df = None    
        self.used_resampling = False

        self.resample_if_ir_below = float(resample_if_ir_below)
        self.resample_if_norm_entropy_below = float(resample_if_norm_entropy_below)

        # dataset stats
        self.imbalance_ratio_pct = None
        self.entropy_bits = None
        self.normalized_entropy = None

    # ─────────────────────────── Data ─────────────────────────── #
    def load_data(self):
        df = pd.read_csv(self.data_path)
        print(f"📂 Loaded dataset with {df.shape[0]} samples")
        if "Diagnosed_PA" not in df.columns:
            raise ValueError("Dataset must contain a 'Diagnosed_PA' column (binary target).")
        X = df.drop(columns=["Diagnosed_PA"])
        y = df["Diagnosed_PA"].astype(int)
        self.feature_names = X.columns.tolist()

        # compute class balance stats
        p_pos = float((y == 1).mean())
        p_min = min(p_pos, 1.0 - p_pos)
        p_max = 1.0 - p_min
        self.imbalance_ratio_pct = 100.0 * (p_min / p_max if p_max > 0 else 0.0)
        self.entropy_bits = _entropy_bits(p_pos)
        self.normalized_entropy = _normalized_entropy_bits(p_pos)

        print(f"📊 Class balance: positive={p_pos:.3f} | IR={self.imbalance_ratio_pct:.1f}% | "
              f"H={self.entropy_bits:.3f} bits (normalized {self.normalized_entropy:.3f})")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Conditional resampling on the TRAIN split only
        if (self.imbalance_ratio_pct < self.resample_if_ir_below or
            self.normalized_entropy < self.resample_if_norm_entropy_below):
            if IMB_OK:
                ros = RandomOverSampler(random_state=42)
                X_res, y_res = ros.fit_resample(self.X_train, self.y_train)
                self.X_train, self.y_train = X_res, y_res
                self.used_resampling = True
                print(f"🧪 Applied RandomOverSampler → train_n={len(self.y_train)} (balanced).")
            else:
                print("⚠️ imbalanced-learn not available; skipping RandomOverSampler.")

    # ────────────────────── Model + GridSearch ─────────────────── #
    def _model_and_grid(self):
        if self.model_type == 'logistic':
            model = LogisticRegression(max_iter=2000, class_weight='balanced', solver="lbfgs")
            param_grid = {'C': [0.1, 1, 10, 100]}
        elif self.model_type == 'rf':
            model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
            param_grid = {'n_estimators': [50, 100, 200, 400], 'max_depth': [5, 10, 20, None]}
        elif self.model_type == 'xgb':
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
                verbosity=0
            )
            param_grid = {
                'n_estimators': [50, 100, 200, 400],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.03, 0.1, 0.2, 0.3],
            }
        elif self.model_type == 'svm':
            model = SVC(probability=True, class_weight='balanced', random_state=42)
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 0.01, 0.1, 1.0], 'kernel': ['rbf']}
        else:
            raise ValueError("Model type must be 'logistic', 'rf', 'xgb', or 'svm'.")
        return model, param_grid

    def train_model(self):
        model, param_grid = self._model_and_grid()

        # Grid search tuned on NEGATIVE log-loss (we’ll flip to positive for plots)
        self.grid_obj = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='neg_log_loss',
            n_jobs=-1,
            refit=True,
            return_train_score=True
        )
        self.grid_obj.fit(self.X_train, self.y_train)
        self.model = self.grid_obj.best_estimator_

        # cv_results
        res = pd.DataFrame(self.grid_obj.cv_results_).copy()
        res["mean_loss"] = res["mean_test_score"].apply(_to_loss)
        res["std_folds"] = res["std_test_score"]  
        res["param_set_index"] = np.arange(len(res))
        self.cv_results_df = res

        print(f"✅ Trained {self.model_type.upper()} model with best params: {self.grid_obj.best_params_}")

    # ────────────────────────── Evaluate ───────────────────────── #
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        print(f"\n📈 Evaluation for {self.model_type.upper()} (held-out test):\n")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        print(f"AUC-ROC: {roc_auc_score(self.y_test, y_prob):.3f}")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.3f}")
        print(f"Precision: {precision_score(self.y_test, y_pred, zero_division=0):.3f}")
        print(f"Recall: {recall_score(self.y_test, y_pred, zero_division=0):.3f}")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        if self.used_resampling:
            print("ℹ️ Training used RandomOverSampler due to imbalance/low entropy.")

    # ───────────── Feature importance / CM / ROC-PR ───────────── #
    def plot_feature_importance(self, save_path=None):
        importances = None
        title = xlabel = None

        if self.model_type == 'logistic':
            importances = self.model.coef_[0]
            title = "Feature Importance (Logistic Regression)"
            xlabel = "Coefficient"
        elif self.model_type in ['rf', 'xgb']:
            importances = self.model.feature_importances_
            title = f"Feature Importance ({self.model_type.upper()})"
            xlabel = "Importance"
        elif self.model_type == 'svm' and hasattr(self.model, 'coef_'):
            importances = self.model.coef_[0]
            title = "Feature Importance (SVM - Linear)"
            xlabel = "Coefficient"
        else:
            return  

        sorted_idx = np.argsort(np.abs(importances))[::-1]
        sorted_features = np.array(self.feature_names)[sorted_idx]
        sorted_importances = np.array(importances)[sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(sorted_features[::-1], sorted_importances[::-1], edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)

        if save_path:
            _ensure_dir(save_path)
            fig.savefig(save_path, bbox_inches="tight")
            print(f"📁 Feature importance plot saved to: {save_path}")
        plt.close(fig)

    def plot_confusion_matrix(self, save_path=None):
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        labels = ["No PA", "Diagnosed PA"]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:d}", ha="center", va="center", fontsize=11)
        ax.set_xticks([0, 1], labels=labels)
        ax.set_yticks([0, 1], labels=labels)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title("Confusion Matrix")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

        if save_path:
            _ensure_dir(save_path)
            fig.savefig(save_path, bbox_inches="tight")
            print(f"📁 Confusion matrix saved to: {save_path}")
        plt.close(fig)

    def plot_roc_pr_curves(self, save_path=None):
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        precision, recall, _ = precision_recall_curve(self.y_test, y_prob)

        fig = plt.figure(figsize=(12, 5))

        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(fpr, tpr, lw=2.0, label=f"AUC = {roc_auc_score(self.y_test, y_prob):.2f}")
        ax1.plot([0, 1], [0, 1], linestyle='--', lw=1.0)
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve")
        ax1.legend()

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(recall, precision, lw=2.0)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")

        if save_path:
            _ensure_dir(save_path)
            fig.savefig(save_path, bbox_inches="tight")
            print(f"📁 ROC & PR curves saved to: {save_path}")
        plt.close(fig)

    # ─────────────────────────── SHAP (robust) ─────────────────────────── #
    def plot_shap_summary(self, save_path=None):
        """Model-specific SHAP with safe fallbacks. Skips if unavailable or errors."""
        if not SHAP_OK:
            print("⚠️ SHAP not available in environment; skipping SHAP plot.")
            return

        try:
            mtype = self.model_type.lower()

            if mtype in ("rf", "xgb"):
                explainer = shap.TreeExplainer(self.model)
                X_eval = self._safe_sample_frame(self.X_test, n=600)
                shap_values = explainer.shap_values(X_eval)
                plt.figure()
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap.summary_plot(shap_values[1], X_eval, show=False)
                else:
                    shap.summary_plot(shap_values, X_eval, show=False)

            elif mtype == "logistic" or (mtype == "svm" and getattr(self.model, "kernel", "linear") == "linear"):
                try:
                    explainer = shap.LinearExplainer(self.model, self.X_train, feature_perturbation="interventional")
                except TypeError:
                    explainer = shap.LinearExplainer(self.model, self.X_train)
                X_eval = self._safe_sample_frame(self.X_test, n=1000)
                shap_values = explainer.shap_values(X_eval)
                plt.figure()
                shap.summary_plot(shap_values, X_eval, show=False)

            elif mtype == "svm":
                # RBF SVM → KernelExplainer on probability function (sampled)
                f = lambda X: self.model.predict_proba(X)[:, 1]
                background = self._safe_sample_frame(self.X_train, n=200)
                explainer = shap.KernelExplainer(f, background, link="identity")
                X_eval = self._safe_sample_frame(self.X_test, n=300)
                shap_values = explainer.shap_values(X_eval, nsamples="auto")
                plt.figure()
                shap.summary_plot(shap_values, X_eval, show=False)

            else:
                print(f"ℹ️ SHAP not configured for model type '{self.model_type}'. Skipping.")
                return

            if save_path:
                _ensure_dir(save_path)
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f"📁 SHAP summary plot saved to: {save_path}")
            plt.close()

        except Exception as e:
            print(f"⚠️ SHAP plotting skipped due to error: {e}")
            try:
                plt.close()
            except Exception:
                pass

    # ───────────── GridSearch diagnostics (per-model panels) ────────── #
    def plot_gridsearch_diagnostics(self, out_dir):
        """
        For each hyperparameter in this model’s GridSearchCV, plot mean log-loss vs value
        with 95% CI ribbons (CI across configs sharing that value).
        Also saves raw cv_results for cross-model aggregation.
        """
        if self.cv_results_df is None:
            return

        df = self.cv_results_df.copy()
        param_cols = [c for c in df.columns if c.startswith("param_")]
        if not param_cols:
            return

        # Save raw cv results (already includes mean_loss)
        raw_path = os.path.join(out_dir, f"gridsearch_results_{self.model_type}.csv")
        _ensure_dir(raw_path)
        df.to_csv(raw_path, index=False)

        # Layout: 2 columns; as many rows as needed
        n_params = len(param_cols)
        ncols = 2
        nrows = int(np.ceil(n_params / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6.6*ncols, 4.9*nrows), squeeze=False, sharey=True
        )

        for idx, pcol in enumerate(param_cols):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]

            series = df[[pcol, "mean_loss"]].copy()
            series.rename(columns={pcol: "value"}, inplace=True)

            # Determine numeric vs categorical axis
            is_numeric = True
            try:
                series["value_num"] = pd.to_numeric(series["value"])
            except Exception:
                is_numeric = False

            grp = series.groupby("value", dropna=False)["mean_loss"].agg(["mean", "std", "count"]).reset_index()
            grp["ci95"] = _ci95(grp["std"].fillna(0.0), grp["count"].clip(lower=1))

            if is_numeric:
                grp = grp.sort_values("value")
                xs = pd.to_numeric(grp["value"]).values
            else:
                xs = np.arange(len(grp))

            ys = grp["mean"].values
            ci = grp["ci95"].values

            # line + shaded CI
            ax.plot(xs, ys, marker="o", lw=2.0, alpha=0.95)
            ax.fill_between(xs, ys - ci, ys + ci, alpha=0.18)

            # labels/ticks and cosmetics
            ax.set_title(f"{self.model_type.upper()} — {pcol.replace('param_','')}")
            ax.set_xlabel(pcol.replace("param_", ""))
            ax.set_ylabel("Log-loss ↓")
            if not is_numeric:
                ax.set_xticks(xs)
                ax.set_xticklabels(grp["value"].astype(str), rotation=0)

            # tidy bounds with a small pad
            ymin = float(np.nanmin(ys - ci))
            ymax = float(np.nanmax(ys + ci))
            pad = 0.06 * (ymax - ymin + 1e-12)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.spines["left"].set_alpha(0.7)
            ax.spines["bottom"].set_alpha(0.7)

        # Hide unused panels
        for k in range(n_params, nrows*ncols):
            r, c = divmod(k, ncols)
            axes[r, c].axis("off")

        out_path = os.path.join(out_dir, f"gridsearch_panels_{self.model_type}.png")
        _ensure_dir(out_path)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"📁 Grid-search panels saved to: {out_path}")
        print(f"📄 Raw grid CSV saved to: {raw_path}")

    # ───────────── Calibration (2×2 grid; one per classifier) ────────── #
    @staticmethod
    def plot_calibration_grid(probas_dict, out_path, n_bins=20, loess_frac=0.25):
        """
        probas_dict: mapping {"LOGISTIC": (y_true, y_prob), "RF": (...), "XGB": (...), "SVM": (...)}
        Saves a 2×2 grid: each panel shows histogram (bottom axis) + reliability curve
        (bin points + LOESS-smoothed curve). Title includes ECE.
        """
        order = ["LOGISTIC", "RF", "XGB", "SVM"]
        present = [m for m in order if m in probas_dict]
        if not present:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), squeeze=False)
        axes = axes.ravel()

        for i, m in enumerate(present[:4]):
            y_true, y_prob = probas_dict[m]
            ax = axes[i]

            # reliability bins
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

            # LOESS smoothing on raw probabilities
            xs, ys = _loess_smooth(prob_pred, prob_true, frac=loess_frac)

            # plot
            ax.plot([0, 1], [0, 1], linestyle="--", lw=1.0, color="#94a3b8", label="Perfect")
            ax.scatter(prob_pred, prob_true, s=26, alpha=0.85, label="Bins")
            ax.plot(xs, ys, lw=2.0, label="LOESS")

            # histogram on twin axis at bottom
            ax_hist = ax.twinx()
            ax_hist.hist(y_prob, bins=20, range=(0, 1), alpha=0.20, edgecolor="none")
            ax_hist.set_yticks([])
            ax_hist.set_ylim(0, None)

            ece = _expected_calibration_error(y_true, y_prob, n_bins=n_bins)
            ax.set_title(f"{m} — Calibration (ECE={ece:.03f})")
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Observed fraction")
            ax.legend(loc="lower right", frameon=False)

        # hide any unused panels
        for k in range(len(present), 4):
            axes[k].axis("off")

        _ensure_dir(out_path)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    # ───────────── PCA + t-SNE (optional 1×2 figure) ────────── #
    def plot_unsupervised_embeddings(self, save_path=None, perplexity=30):
        """Make a 1×2 figure: PCA(2D) and t-SNE(2D), colored by true label (diagnostic)."""
        if not isinstance(self.X_train, (pd.DataFrame, pd.Series, np.ndarray)):
            return
        X_full = pd.concat([self.X_train, self.X_test], axis=0)
        y_full = pd.concat([self.y_train, self.y_test], axis=0).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_full)

        pca = PCA(n_components=2, random_state=7)
        X_pca = pca.fit_transform(X_scaled)
        ev = pca.explained_variance_ratio_
        pca_title = f"PCA (2D) — var: {100*ev[0]:.1f}% + {100*ev[1]:.1f}%"

        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=7)
        X_tsne = tsne.fit_transform(X_scaled)

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), squeeze=False)
        ax1, ax2 = axes[0, 0], axes[0, 1]
        colors = {0: "#4464ad", 1: "#e07a5f"}
        labels = {0: "Control (0)", 1: "PA (1)"}

        for cls in [0, 1]:
            mask = (y_full == cls)
            ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], s=20, alpha=0.70, c=colors[cls], label=labels[cls])
            ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=20, alpha=0.70, c=colors[cls], label=labels[cls])

        ax1.set_title(pca_title); ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.legend(frameon=False)
        ax2.set_title("t-SNE (2D) — unsupervised; PCA init"); ax2.set_xlabel("Dim 1"); ax2.set_ylabel("Dim 2"); ax2.legend(frameon=False)
        for ax in (ax1, ax2):
            ax.spines["left"].set_alpha(0.7); ax.spines["bottom"].set_alpha(0.7); ax.grid(True, alpha=0.25, linestyle="--")

        if save_path:
            _ensure_dir(save_path)
            fig.savefig(save_path)
            print(f"🖼️ PCA + t-SNE figure saved to: {save_path}")
        plt.close(fig)

# ───────────────────────────── CLI ───────────────────────────── #
def main():
    parser = argparse.ArgumentParser(
        description="Train ML model for Pernicious Anaemia detection (with grid diagnostics, calibration, and optional resampling)."
    )
    parser.add_argument("--data", type=str, required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--model", type=str, choices=["logistic", "rf", "xgb", "svm"],
                        default="logistic", help="Model type.")
    parser.add_argument("--savefigs", action="store_true", help="Save plots to disk.")
    parser.add_argument("--output_model", type=str, default="output/model.pkl", help="Path to save trained model.")
    parser.add_argument("--output_figs_dir", type=str, default="output/figs", help="Directory to save plots.")
    parser.add_argument("--resample_ir_below", type=float, default=40.0,
                        help="Apply RandomOverSampler if IR%% below this threshold.")
    parser.add_argument("--resample_normH_below", type=float, default=0.65,
                        help="Apply RandomOverSampler if normalized entropy below this threshold.")
    args = parser.parse_args()

    trainer = PAModelTrainer(
        data_path=args.data,
        model_type=args.model,
        resample_if_ir_below=args.resample_ir_below,
        resample_if_norm_entropy_below=args.resample_normH_below
    )
    trainer.load_data()
    trainer.train_model()
    trainer.evaluate_model()

    _ensure_dir(args.output_model)
    joblib.dump(trainer.model, args.output_model)
    print(f"🧠 Model saved to: {args.output_model}")

    # Always save grid diagnostics so your all-models overview can use the CSVs
    trainer.plot_gridsearch_diagnostics(out_dir=args.output_figs_dir)

    if args.savefigs:
        trainer.plot_feature_importance(save_path=f"{args.output_figs_dir}/feature_importance_{args.model}.png")
        trainer.plot_confusion_matrix(save_path=f"{args.output_figs_dir}/confusion_matrix_{args.model}.png")
        trainer.plot_roc_pr_curves(save_path=f"{args.output_figs_dir}/roc_pr_curve_{args.model}.png")
        trainer.plot_shap_summary(save_path=f"{args.output_figs_dir}/shap_summary_{args.model}.png")
        # Optional: embeddings figure (for EDA/appendix)
        trainer.plot_unsupervised_embeddings(save_path=f"{args.output_figs_dir}/embeddings_pca_tsne.png")

if __name__ == "__main__":
    main()
