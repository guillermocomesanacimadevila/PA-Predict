#!/usr/bin/env python
# Scripts/pa_model_trainer.py

import argparse
import os
import joblib
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend("Agg")  # headless-safe for CI/conda run

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Optional SHAP
try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

# ───────────────── Aesthetics (publishable) ───────────────── #
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.constrained_layout.use": False,   # keep one engine; avoid colorbar clash
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

# ───────────────── Trainer ───────────────── #
class PAModelTrainer:
    # ─────────────── Helpers kept inside the class for portability ─────────────── #
    @staticmethod
    def _ensure_dir(p: str):
        d = os.path.dirname(p) if os.path.splitext(p)[1] else p
        if d:
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def _to_loss(neg_score):
        # GridSearchCV uses neg_* scorers for losses; flip sign to positive loss
        return -float(neg_score)

    @staticmethod
    def _ci95(std, n):
        std = np.asarray(std, dtype=float)
        n = np.asarray(n, dtype=float)
        return 1.96 * (std / np.sqrt(np.maximum(1.0, n)))

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
    def __init__(self, data_path, model_type='logistic'):
        self.data_path = data_path
        self.model_type = model_type
        self.model = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.feature_names = None
        self.grid_obj = None         # GridSearchCV object
        self.cv_results_df = None    # tidy cv results

    # ─────────────────────────── Data ─────────────────────────── #
    def load_data(self):
        df = pd.read_csv(self.data_path)
        print(f"📂 Loaded dataset with {df.shape[0]} samples")
        X = df.drop(columns=["Diagnosed_PA"])
        y = df["Diagnosed_PA"]
        self.feature_names = X.columns.tolist()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    # ────────────────────── Model + GridSearch ─────────────────── #
    def _model_and_grid(self):
        if self.model_type == 'logistic':
            model = LogisticRegression(max_iter=2000, class_weight='balanced', solver="lbfgs")
            param_grid = {'C': [0.1, 1, 10, 100]}
        elif self.model_type == 'rf':
            model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [50, 100, 200, 400],
                'max_depth': [5, 10, 20, None]
            }
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

        # Use NEGATIVE log-loss to tune; we plot positive loss.
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

        # Tidy cv_results
        res = pd.DataFrame(self.grid_obj.cv_results_).copy()
        res["mean_loss"] = res["mean_test_score"].apply(self._to_loss)
        res["std_folds"] = res["std_test_score"]  # std across folds
        res["param_set_index"] = np.arange(len(res))
        self.cv_results_df = res

        print(f"✅ Trained {self.model_type.upper()} model with best params: {self.grid_obj.best_params_}")

    # ────────────────────────── Evaluate ───────────────────────── #
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        # FIX: .upper() (correct) instead of .UPPER()
        print(f"\n📈 Evaluation for {self.model_type.upper()} (held-out test):\n")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        print(f"AUC-ROC: {roc_auc_score(self.y_test, y_prob):.3f}")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.3f}")
        print(f"Precision: {precision_score(self.y_test, y_pred, zero_division=0):.3f}")
        print(f"Recall: {recall_score(self.y_test, y_pred, zero_division=0):.3f}")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))

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
            return  # e.g., non-linear SVM

        sorted_idx = np.argsort(np.abs(importances))[::-1]
        sorted_features = np.array(self.feature_names)[sorted_idx]
        sorted_importances = np.array(importances)[sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(sorted_features[::-1], sorted_importances[::-1], edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)

        if save_path:
            self._ensure_dir(save_path)
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
            self._ensure_dir(save_path)
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
            self._ensure_dir(save_path)
            fig.savefig(save_path, bbox_inches="tight")
            print(f"📁 ROC & PR curves saved to: {save_path}")
        plt.close(fig)

    # ─────────────────────────── SHAP (robust) ─────────────────────────── #
    def plot_shap_summary(self, save_path=None):
        """Robust SHAP summary plot with model-specific explainers.
           Falls back gracefully if SHAP is unavailable or errors occur.
        """
        try:
            import shap
        except Exception:
            print("⚠️ SHAP not available in environment; skipping SHAP plot.")
            return

        try:
            mtype = self.model_type.lower()

            if mtype in ("rf", "xgb"):
                explainer = shap.TreeExplainer(self.model)
                X_eval = self._safe_sample_frame(self.X_test, n=600)
                shap_values = explainer.shap_values(X_eval)
                plt.figure()
                # RF returns [class0, class1] sometimes
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
                # RBF SVM → KernelExplainer on probability function
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
                self._ensure_dir(save_path)
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                print(f"📁 SHAP summary plot saved to: {save_path}")
            plt.close()

        except Exception as e:
            print(f"⚠️ SHAP plotting skipped due to error: {e}")
            try:
                plt.close()
            except Exception:
                pass

    # ───────────── GridSearch diagnostics (multi-panel) ────────── #
    def plot_gridsearch_diagnostics(self, out_dir):
        """
        Multi-panel grid-search diagnostics (per hyperparameter) with
        mean log-loss lines and 95% CI ribbons (CI across other-param settings).
        Also saves the raw cv_results for possible cross-model aggregation.
        """
        if self.cv_results_df is None:
            return

        df = self.cv_results_df.copy()
        param_cols = [c for c in df.columns if c.startswith("param_")]
        if not param_cols:
            return

        # Save raw cv results (already includes mean_loss)
        raw_path = os.path.join(out_dir, f"gridsearch_results_{self.model_type}.csv")
        self._ensure_dir(raw_path)
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
            grp["ci95"] = self._ci95(grp["std"].fillna(0.0), grp["count"].clip(lower=1))

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
        self._ensure_dir(out_path)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"📁 Grid-search panels saved to: {out_path}")
        print(f"📄 Raw grid CSV saved to: {raw_path}")

# ───────────────────────────── CLI ───────────────────────────── #
def main():
    parser = argparse.ArgumentParser(
        description="Train ML model for Pernicious Anaemia detection (with grid diagnostics)."
    )
    parser.add_argument("--data", type=str, required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--model", type=str, choices=["logistic", "rf", "xgb", "svm"],
                        default="logistic", help="Model type.")
    parser.add_argument("--savefigs", action="store_true", help="Save plots to disk.")
    parser.add_argument("--output_model", type=str, default="output/model.pkl", help="Path to save trained model.")
    parser.add_argument("--output_figs_dir", type=str, default="output/figs", help="Directory to save plots.")
    args = parser.parse_args()

    trainer = PAModelTrainer(data_path=args.data, model_type=args.model)
    trainer.load_data()
    trainer.train_model()
    trainer.evaluate_model()

    _ensure_dir(args.output_model)
    joblib.dump(trainer.model, args.output_model)
    print(f"🧠 Model saved to: {args.output_model}")

    # Always save grid diagnostics so your “all-models” plot can be built later
    trainer.plot_gridsearch_diagnostics(out_dir=args.output_figs_dir)

    if args.savefigs:
        trainer.plot_feature_importance(save_path=f"{args.output_figs_dir}/feature_importance_{args.model}.png")
        trainer.plot_confusion_matrix(save_path=f"{args.output_figs_dir}/confusion_matrix_{args.model}.png")
        trainer.plot_roc_pr_curves(save_path=f"{args.output_figs_dir}/roc_pr_curve_{args.model}.png")
        trainer.plot_shap_summary(save_path=f"{args.output_figs_dir}/shap_summary_{args.model}.png")
    # If not saving, grid panels are already emitted above.

if __name__ == "__main__":
    main()
