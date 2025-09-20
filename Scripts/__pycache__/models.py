#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pernicious Anaemia (PA) robust trainer:
Models: xgb, rf, logistic, lightgbm, svm, torch_mlp
- Pipeline with ColumnTransformer (impute+scale numeric, OHE categorical)
- RandomizedSearchCV with PR-AUC refit; repeated stratified CV or stratified-group CV
- Probability calibration (isotonic) for sklearn/xgb/lgbm/svm models
- Threshold tuning (optionally enforce minimum precision)
- Metrics with 95% bootstrap CIs; plots: ROC, PR, Confusion, Calibration, Learning Curve
- Subgroup metrics
- SHAP summary for tree models (XGB/RF/LightGBM)
- Saves model, threshold, and run metadata
"""

import argparse, json, os, sys, warnings
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score, fbeta_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve, make_scorer)
from sklearn.model_selection import (RepeatedStratifiedKFold, RandomizedSearchCV,
                                     StratifiedKFold, learning_curve, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Optional LightGBM (handled gracefully if missing)
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# Optional PyTorch (handled gracefully if missing)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


@dataclass
class RunMetadata:
    timestamp: str
    model_type: str
    best_params: dict
    chosen_threshold: float
    train_size: int
    test_size: int
    pos_rate_train: float
    pos_rate_test: float
    metrics: dict
    sklearn_version: str
    xgboost_version: str
    lightgbm_version: str
    shap_version: str
    torch_version: str
    python_version: str
    feature_names: list
    expanded_feature_names: list


# ---------- Simple sklearn-compatible PyTorch MLP ----------
class TorchMLPClassifier:
    """Minimal sklearn-compatible MLP with internal val split + early stopping.
       Returns calibrated probabilities via sigmoid on logits.
    """
    def __init__(self, input_dim=None, hidden_dim=64, hidden_layers=1, dropout=0.2,
                 lr=1e-3, weight_decay=1e-4, batch_size=256, epochs=50,
                 patience=5, random_state=RANDOM_STATE, device="auto"):
        if not HAS_TORCH:
            raise ImportError("PyTorch not available.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.device = device
        self._fitted = False

    def _build(self, in_dim):
        layers = []
        dim = in_dim
        for _ in range(self.hidden_layers):
            layers += [nn.Linear(dim, self.hidden_dim), nn.ReLU(), nn.Dropout(self.dropout)]
            dim = self.hidden_dim
        layers += [nn.Linear(dim, 1)]  # logits
        self.model = nn.Sequential(*layers)

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n = X.shape[0]
        # 80/20 internal validation split
        idx = rng.permutation(n)
        n_val = max(1, int(0.2 * n))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        in_dim = X.shape[1] if self.input_dim is None else self.input_dim
        self._build(in_dim)
        dev = torch.device("cuda:0" if (self.device == "auto" and torch.cuda.is_available()) else "cpu")
        self.model.to(dev)

        ds_tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        ds_val = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(yval))
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=False)
        dl_val = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False, drop_last=False)

        criterion = nn.BCEWithLogitsLoss()
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_state = None
        best_loss = np.inf
        patience_ctr = 0

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

            # val
            self.model.eval()
            with torch.no_grad():
                vloss = 0.0
                n_batches = 0
                for xb, yb in dl_val:
                    xb, yb = xb.to(dev), yb.to(dev)
                    logits = self.model(xb)
                    vloss += criterion(logits, yb).item()
                    n_batches += 1
                vloss /= max(1, n_batches)

            if vloss < best_loss - 1e-4:
                best_loss = vloss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.device_used_ = str(dev)
        self._fitted = True
        return self

    def _predict_proba_on(self, X):
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=np.float32)
        dev = torch.device(self.device_used_)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).to(dev)).cpu().numpy().reshape(-1)
            prob1 = 1.0 / (1.0 + np.exp(-logits))
        prob0 = 1.0 - prob1
        return np.vstack([prob0, prob1]).T

    def predict_proba(self, X):
        return self._predict_proba_on(X)

    def decision_function(self, X):
        # For potential external calibration (not used here)
        X = np.asarray(X, dtype=np.float32)
        dev = torch.device(self.device_used_)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).to(dev)).cpu().numpy().reshape(-1)
        return logits

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    # sklearn API
    def get_params(self, deep=True):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "random_state": self.random_state,
            "device": self.device,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class Trainer:
    def __init__(self, data_path, model_type="xgb", target_col="Diagnosed_PA",
                 patient_id_col=None, subgroup_cols=None, min_precision=None,
                 test_size=0.2, output_dir="output", savefigs=True):
        self.data_path = data_path
        self.model_type = model_type
        self.target_col = target_col
        self.patient_id_col = patient_id_col
        self.subgroup_cols = subgroup_cols or []
        self.min_precision = min_precision
        self.test_size = test_size
        self.output_dir = output_dir
        self.savefigs = savefigs
        self.fig_dir = os.path.join(output_dir, "figs") if savefigs else None
        os.makedirs(self.output_dir, exist_ok=True)
        if self.savefigs:
            os.makedirs(self.fig_dir, exist_ok=True)

        self.model = None
        self.pre = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.groups = None
        self.feature_names = None
        self.expanded_feature_names = None
        self.best_params_ = {}
        self.chosen_threshold_ = 0.5

    # -------- Data --------
    def load_data(self):
        df = pd.read_csv(self.data_path)
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found.")
        y = df[self.target_col].astype(int).values

        if self.patient_id_col and self.patient_id_col in df.columns:
            self.groups = df[self.patient_id_col].astype(str).values
            X = df.drop(columns=[self.target_col, self.patient_id_col])
        else:
            X = df.drop(columns=[self.target_col])

        self.feature_names = X.columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in self.feature_names if c not in num_cols]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=RANDOM_STATE
        )

        self.pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler(with_mean=False))]), num_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))]), cat_cols)
        ])

        print(f"📂 Loaded {df.shape[0]} rows, {X.shape[1]} features.")
        print(f"🔎 Train={self.X_train.shape[0]} (pos {self.y_train.mean():.3f}) "
              f"Test={self.X_test.shape[0]} (pos {self.y_test.mean():.3f})")

    # -------- Models --------
    def _make_base(self):
        m = self.model_type.lower()
        if m == "logistic":
            return LogisticRegression(solver="saga", penalty="l2", max_iter=5000,
                                      class_weight="balanced", random_state=RANDOM_STATE)
        if m == "rf":
            return RandomForestClassifier(n_estimators=400, max_depth=None,
                                          class_weight="balanced", n_jobs=-1,
                                          random_state=RANDOM_STATE)
        if m == "xgb":
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=4,
                                 subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                                 objective="binary:logistic", eval_metric="logloss",
                                 n_jobs=-1, random_state=RANDOM_STATE)
        if m == "svm":
            # Wrap with calibration later to get probabilities
            return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                       random_state=RANDOM_STATE, probability=False)
        if m == "lightgbm":
            if not HAS_LGBM:
                raise ImportError("LightGBM not installed. pip install lightgbm")
            return LGBMClassifier(n_estimators=600, learning_rate=0.05, max_depth=-1,
                                  subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                                  objective="binary", n_jobs=-1, random_state=RANDOM_STATE)
        if m == "torch_mlp":
            if not HAS_TORCH:
                raise ImportError("PyTorch not installed.")
            # Torch model will be used directly (no external calibration here)
            return TorchMLPClassifier()
        raise ValueError("Unknown model type.")

    def _search_space(self, prefix="clf"):
        m = self.model_type.lower()
        if m == "logistic":
            return {f"{prefix}__base_estimator__C": np.logspace(-3, 1, 10)}
        if m == "rf":
            return {
                f"{prefix}__base_estimator__n_estimators": np.arange(200, 1201, 100),
                f"{prefix}__base_estimator__max_depth": [None, 6, 8, 10, 12],
                f"{prefix}__base_estimator__min_samples_split": [2, 5, 10],
                f"{prefix}__base_estimator__min_samples_leaf": [1, 2, 4],
                f"{prefix}__base_estimator__max_features": ["sqrt", "log2", None],
            }
        if m == "xgb":
            return {
                f"{prefix}__base_estimator__n_estimators": np.arange(300, 1501, 100),
                f"{prefix}__base_estimator__max_depth": [3, 4, 5, 6],
                f"{prefix}__base_estimator__learning_rate": np.logspace(-3, -1, 7),
                f"{prefix}__base_estimator__subsample": [0.7, 0.8, 0.9, 1.0],
                f"{prefix}__base_estimator__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                f"{prefix}__base_estimator__reg_lambda": np.logspace(-3, 2, 6),
                f"{prefix}__base_estimator__scale_pos_weight": [1, 2, 3, 5, 8, 12],
            }
        if m == "svm":
            return {
                f"{prefix}__base_estimator__C": np.logspace(-2, 2, 10),
                f"{prefix}__base_estimator__kernel": ["linear", "rbf"],
                f"{prefix}__base_estimator__gamma": ["scale", "auto"],
            }
        if m == "lightgbm":
            # Only if installed
            return {
                f"{prefix}__base_estimator__n_estimators": np.arange(300, 1501, 100),
                f"{prefix}__base_estimator__learning_rate": np.logspace(-3, -1, 7),
                f"{prefix}__base_estimator__num_leaves": [15, 31, 63, 127],
                f"{prefix}__base_estimator__min_child_samples": [5, 10, 20, 40],
                f"{prefix}__base_estimator__subsample": [0.7, 0.8, 0.9, 1.0],
                f"{prefix}__base_estimator__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                f"{prefix}__base_estimator__reg_lambda": np.logspace(-3, 2, 6),
            }
        if m == "torch_mlp":
            # Torch model tuned on its own params; calibration not applied externally
            return {
                f"{prefix}__hidden_dim": [32, 64, 128],
                f"{prefix}__hidden_layers": [1, 2, 3],
                f"{prefix}__dropout": [0.0, 0.2, 0.4],
                f"{prefix}__lr": np.logspace(-4, -2, 5),
                f"{prefix}__weight_decay": np.logspace(-6, -3, 4),
                f"{prefix}__batch_size": [128, 256, 512],
                f"{prefix}__epochs": [30, 50, 80],
                f"{prefix}__patience": [3, 5, 8],
            }
        return {}

    # -------- Train --------
    def train(self):
        base = self._make_base()

        # For non-torch models, wrap in isotonic calibration for calibrated probs
        if self.model_type.lower() != "torch_mlp":
            clf = CalibratedClassifierCV(base_estimator=base, method="isotonic", cv=3)
        else:
            clf = base  # torch model already outputs calibrated-like sigmoid probs

        pipe = Pipeline([("pre", self.pre), ("clf", clf)])

        # CV strategy
        if self.groups is not None:
            try:
                from sklearn.model_selection import StratifiedGroupKFold
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                cv_for_lc = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                fit_params = {"clf__cv": 3} if self.model_type.lower() != "torch_mlp" else {}
                fit_extra = {"groups": self.groups[: len(self.X_train)]}
            except Exception:
                print("⚠️ StratifiedGroupKFold unavailable; using RepeatedStratifiedKFold.")
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
                cv_for_lc = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                fit_params, fit_extra = {}, {}
        else:
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
            cv_for_lc = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            fit_params, fit_extra = {}, {}

        scorers = {
            "pr_auc": make_scorer(average_precision_score, needs_proba=True),
            "roc_auc": "roc_auc",
            "recall": "recall",
            "f2": make_scorer(fbeta_score, beta=2),
        }

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=self._search_space("clf"),
            n_iter=40,
            scoring=scorers,
            refit="pr_auc",
            cv=cv,
            n_jobs=-1 if self.model_type.lower() != "torch_mlp" else 1,
            random_state=RANDOM_STATE,
            verbose=1,
        )

        search.fit(self.X_train, self.y_train, **fit_params, **fit_extra)
        self.model = search.best_estimator_
        self.best_params_ = search.best_params_
        print("✅ Best params:", self.best_params_)

        # Expanded feature names
        self.expanded_feature_names = self._expanded_feature_names(self.model.named_steps["pre"])

        # Learning curve
        self._plot_learning_curve(self.model, self.X_train, self.y_train, cv_for_lc)

    # -------- Evaluate --------
    def evaluate(self):
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        self.chosen_threshold_ = self._choose_threshold(self.y_test, y_prob, self.min_precision)
        y_pred = (y_prob >= self.chosen_threshold_).astype(int)

        print("\n📈 Report @ chosen threshold")
        print(classification_report(self.y_test, y_pred, zero_division=0))

        metrics = {
            "roc_auc": float(roc_auc_score(self.y_test, y_prob)),
            "pr_auc": float(average_precision_score(self.y_test, y_prob)),
            "brier": float(brier_score_loss(self.y_test, y_prob)),
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(self.y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(self.y_test, y_pred, zero_division=0)),
        }
        ci = {
            "roc_auc_ci": self._ci_bootstrap(lambda yt, yp: roc_auc_score(yt, yp), self.y_test, y_prob),
            "pr_auc_ci": self._ci_bootstrap(lambda yt, yp: average_precision_score(yt, yp), self.y_test, y_prob),
            "recall_ci": self._ci_bootstrap(
                lambda yt, yp: recall_score(yt, (yp >= self.chosen_threshold_).astype(int), zero_division=0),
                self.y_test, y_prob),
        }
        print(f"AUC-ROC {metrics['roc_auc']:.3f} (95% CI {ci['roc_auc_ci'][0]:.3f}-{ci['roc_auc_ci'][1]:.3f})")
        print(f"AUC-PR  {metrics['pr_auc']:.3f} (95% CI {ci['pr_auc_ci'][0]:.3f}-{ci['pr_auc_ci'][1]:.3f})")
        print(f"Brier   {metrics['brier']:.3f}")
        print(f"Threshold: {self.chosen_threshold_:.3f}")

        # Plots
        self._plot_roc_pr(self.y_test, y_prob)
        self._plot_confusion(self.y_test, y_pred)
        self._plot_calibration(self.y_test, y_prob)

        # Subgroups
        sub = self._subgroup_metrics()

        # Persist
        self._save_all(metrics, ci, sub)

    # -------- Helpers --------
    def _expanded_feature_names(self, pre):
        num_feats, cat_feats = [], []
        for name, trans, cols in pre.transformers_:
            if name == "num":
                num_feats = list(cols)
            elif name == "cat":
                ohe = trans.named_steps["ohe"]
                cat_feats = ohe.get_feature_names_out(cols).tolist()
        return num_feats + cat_feats

    def _choose_threshold(self, y_true, y_prob, min_precision=None):
        if min_precision is None:
            p, r, t = precision_recall_curve(y_true, y_prob)
            f2 = (5 * p * r) / (4 * p + r + 1e-12)
            i = int(np.nanargmax(f2))
            return float(t[max(i - 1, 0)]) if i < len(t) else 0.5
        p, r, t = precision_recall_curve(y_true, y_prob)
        mask = p[:-1] >= min_precision
        if not np.any(mask):
            print(f"⚠️ No threshold achieves precision ≥ {min_precision:.2f}; using 0.5")
            return 0.5
        idxs = np.where(mask)[0]
        best = idxs[np.argmax(r[idxs])]
        return float(t[max(best, 0)])

    def _ci_bootstrap(self, fn, y_true, y_prob, n=2000, seed=RANDOM_STATE):
        rng = np.random.default_rng(seed)
        stats = []
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        for _ in range(n):
            idx = rng.integers(0, len(y_true), len(y_true))
            stats.append(fn(y_true[idx], y_prob[idx]))
        low, high = np.percentile(stats, [2.5, 97.5])
        return float(low), float(high)

    # -------- Plots --------
    def _save_or_show(self, fig, name):
        if not self.savefigs:
            plt.show()
            return
        path = os.path.join(self.fig_dir, name)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"📁 Saved: {path}")

    def _plot_roc_pr(self, y, prob):
        fpr, tpr, _ = roc_curve(y, prob)
        fig = plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, prob):.2f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend()
        self._save_or_show(fig, "roc_curve.png")

        pr, rc, _ = precision_recall_curve(y, prob)
        fig = plt.figure(figsize=(6,5))
        plt.plot(rc, pr, label=f"AP={average_precision_score(y, prob):.2f}")
        plt.hlines(y.mean(), 0, 1, linestyles="--", label=f"Baseline={y.mean():.2f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.legend()
        self._save_or_show(fig, "pr_curve.png")

    def _plot_confusion(self, y, yhat):
        cm = confusion_matrix(y, yhat)
        fig = plt.figure(figsize=(5.5,5))
        im = plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix"); plt.colorbar(im, fraction=0.046, pad=0.04)
        ticks = np.arange(2); classes = ["No PA", "Diagnosed PA"]
        plt.xticks(ticks, classes, rotation=45); plt.yticks(ticks, classes)
        thresh = cm.max()/2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i,j]:d}", ha="center",
                         color="white" if cm[i,j] > thresh else "black")
        plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.tight_layout()
        self._save_or_show(fig, "confusion_matrix.png")

    def _plot_calibration(self, y, prob, n_bins=10):
        obs, pred = calibration_curve(y, prob, n_bins=n_bins, strategy="quantile")
        fig = plt.figure(figsize=(6,5))
        plt.plot(pred, obs, marker="o", label="Calibrated")
        plt.plot([0,1],[0,1],'--', label="Perfect")
        plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
        plt.title("Calibration"); plt.legend()
        self._save_or_show(fig, "calibration_curve.png")

    def _plot_learning_curve(self, estimator, X, y, cv):
        ts, tr, va = learning_curve(estimator, X, y, cv=cv, scoring="roc_auc",
                                    n_jobs=-1 if self.model_type.lower()!="torch_mlp" else 1,
                                    train_sizes=np.linspace(0.1,1.0,10))
        fig = plt.figure(figsize=(6.5,5))
        plt.plot(ts, tr.mean(axis=1), "o-", label="Training")
        plt.plot(ts, va.mean(axis=1), "o-", label="Validation")
        plt.xlabel("Training size"); plt.ylabel("ROC-AUC"); plt.title("Learning Curve")
        plt.legend(); plt.grid(alpha=0.3)
        self._save_or_show(fig, "learning_curve.png")

    # -------- Subgroups + SHAP --------
    def _subgroup_metrics(self):
        if not self.subgroup_cols:
            return {}
        df = self.X_test.copy()
        df[self.target_col] = self.y_test
        prob = self.model.predict_proba(self.X_test)[:,1]
        out = {}
        for col in self.subgroup_cols:
            if col not in df.columns: 
                print(f"ℹ️ Subgroup '{col}' not found; skipping.")
                continue
            for lvl, idx in df.groupby(col).groups.items():
                idx = np.array(list(idx))
                if idx.size < 10: 
                    continue
                yt = self.y_test[idx]; yp = prob[idx]
                yhat = (yp >= self.chosen_threshold_).astype(int)
                out[f"{col}={lvl}"] = {
                    "n": int(idx.size),
                    "pos_rate": float(yt.mean()),
                    "roc_auc": float(roc_auc_score(yt, yp)) if len(np.unique(yt))>1 else np.nan,
                    "pr_auc": float(average_precision_score(yt, yp)),
                    "precision": float(precision_score(yt, yhat, zero_division=0)),
                    "recall": float(recall_score(yt, yhat, zero_division=0)),
                }
        if out:
            print("\n📊 Subgroup metrics:")
            for k,v in out.items(): print(f"- {k}: {v}")
        # SHAP only for tree models
        self._plot_shap_tree_models()
        return out

    def _plot_shap_tree_models(self):
        m = self.model_type.lower()
        if m not in {"xgb","rf","lightgbm"}:
            return
        try:
            pre = self.model.named_steps["pre"]
            X_tr = pre.transform(self.X_train)
            X_te = pre.transform(self.X_test)
            clf = self.model.named_steps["clf"]
            base = getattr(clf, "base_estimator", clf)
            # Choose explainer
            explainer = shap.TreeExplainer(base)
            shap_vals = explainer(X_te)
            fig = plt.figure()
            shap.summary_plot(shap_vals, X_te, feature_names=self.expanded_feature_names,
                              show=False, plot_type="dot", max_display=25)
            self._save_or_show(fig, f"shap_summary_{m}.png")
        except Exception as e:
            print(f"⚠️ SHAP failed: {e}")

    # -------- Save --------
    def _save_all(self, metrics, ci, sub):
        model_path = os.path.join(self.output_dir, f"model_{self.model_type}.pkl")
        joblib.dump(self.model, model_path)
        with open(os.path.join(self.output_dir, "threshold.json"), "w") as f:
            json.dump({"threshold": self.chosen_threshold_}, f, indent=2)

        meta = RunMetadata(
            timestamp=datetime.utcnow().isoformat()+"Z",
            model_type=self.model_type,
            best_params=self.best_params_,
            chosen_threshold=self.chosen_threshold_,
            train_size=int(self.X_train.shape[0]),
            test_size=int(self.X_test.shape[0]),
            pos_rate_train=float(self.y_train.mean()),
            pos_rate_test=float(self.y_test.mean()),
            metrics={**metrics, **ci},
            sklearn_version=self._ver("sklearn"),
            xgboost_version=self._ver("xgboost"),
            lightgbm_version=self._ver("lightgbm") if HAS_LGBM else "not_installed",
            shap_version=self._ver("shap"),
            torch_version=self._ver("torch") if HAS_TORCH else "not_installed",
            python_version=sys.version.replace("\n"," "),
            feature_names=self.feature_names,
            expanded_feature_names=self.expanded_feature_names or [],
        )
        with open(os.path.join(self.output_dir, "run_metadata.json"), "w") as f:
            json.dump(asdict(meta), f, indent=2)

        print(f"🧠 Model saved: {model_path}")
        print(f"🧾 Metadata: {os.path.join(self.output_dir, 'run_metadata.json')}")

    @staticmethod
    def _ver(pkg):
        try:
            mod = __import__(pkg)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return "unknown"


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Robust PA trainer with multiple models.")
    p.add_argument("--data", type=str, required=True, help="Path to CSV dataset.")
    p.add_argument("--target", type=str, default="Diagnosed_PA", help="Target column.")
    p.add_argument("--model", type=str,
                   choices=["xgb","rf","logistic","lightgbm","svm","torch_mlp"],
                   default="xgb")
    p.add_argument("--patient_id", type=str, default=None, help="Optional patient_id column for grouped CV.")
    p.add_argument("--subgroups", type=str, default=None,
                   help="Comma-separated feature cols for subgroup metrics (e.g., sex,site,age_band).")
    p.add_argument("--min_precision", type=float, default=None,
                   help="If set (e.g., 0.8), choose threshold with precision ≥ this value.")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--output_dir", type=str, default="output")
    p.add_argument("--no_figs", action="store_true", help="Disable saving plots.")
    return p.parse_args()


def main():
    args = parse_args()
    subgroup_cols = [s.strip() for s in args.subgroups.split(",")] if args.subgroups else []

    t = Trainer(
        data_path=args.data,
        model_type=args.model,
        target_col=args.target,
        patient_id_col=args.patient_id,
        subgroup_cols=subgroup_cols,
        min_precision=args.min_precision,
        test_size=args.test_size,
        output_dir=args.output_dir,
        savefigs=not args.no_figs,
    )
    t.load_data()
    t.train()
    t.evaluate()


if __name__ == "__main__":
    main()
