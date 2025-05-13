import pandas as pd
import numpy as np
import argparse
import os
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

class PAModelTrainer:
    def __init__(self, data_path, model_type='logistic'):
        self.data_path = data_path
        self.model_type = model_type
        self.model = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.feature_names = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        print(f"📂 Loaded dataset with {df.shape[0]} samples")
        X = df.drop(columns=["Diagnosed_PA"])
        y = df["Diagnosed_PA"]
        self.feature_names = X.columns.tolist()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    def train_model(self):
        if self.model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
            param_grid = {'C': [0.01, 0.1, 1, 10]}
        elif self.model_type == 'rf':
            model = RandomForestClassifier(class_weight='balanced', random_state=42)
            param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        elif self.model_type == 'xgb':
            model = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=5,
                random_state=42
            )
            param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7]}
        else:
            raise ValueError("Model type must be 'logistic', 'rf', or 'xgb'.")

        grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        self.model = grid.best_estimator_
        print(f"✅ Trained {self.model_type.upper()} model with best params: {grid.best_params_}")

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        print(f"\n📈 Evaluation for {self.model_type.upper()}:\n")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        print(f"AUC-ROC: {roc_auc_score(self.y_test, y_prob):.3f}")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.3f}")
        print(f"Precision: {precision_score(self.y_test, y_pred, zero_division=0):.3f}")
        print(f"Recall: {recall_score(self.y_test, y_pred, zero_division=0):.3f}")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))

    def plot_feature_importance(self, save_path=None):
        if self.model_type == 'logistic':
            importances = self.model.coef_[0]
            title = "Feature Importance (Logistic Regression)"
            xlabel = "Coefficient"
        elif self.model_type in ['rf', 'xgb']:
            importances = self.model.feature_importances_
            title = f"Feature Importance ({self.model_type.upper()})"
            xlabel = "Importance"
        else:
            return

        sorted_idx = np.argsort(np.abs(importances))[::-1]
        sorted_features = np.array(self.feature_names)[sorted_idx]
        sorted_importances = np.array(importances)[sorted_idx]

        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid", font_scale=1.2)
        sns.barplot(x=sorted_importances, y=sorted_features, palette="Blues_d", edgecolor="black")
        plt.title(title, fontsize=16, weight='bold')
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel("Features", fontsize=14)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"📁 Feature importance plot saved to: {save_path}")
        else:
            plt.show()

    def plot_confusion_matrix(self, save_path=None):
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        labels = ["No PA", "Diagnosed PA"]

        plt.figure(figsize=(6, 5))
        sns.set(style="white", font_scale=1.2)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.5,
            cbar=False
        )
        plt.ylabel("Actual", fontsize=14)
        plt.xlabel("Predicted", fontsize=14)
        plt.title("Confusion Matrix", fontsize=16, weight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"📁 Confusion matrix saved to: {save_path}")
        else:
            plt.show()

    def plot_roc_pr_curves(self, save_path=None):
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        precision, recall, _ = precision_recall_curve(self.y_test, y_prob)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(self.y_test, y_prob):.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"📁 ROC & PR curves saved to: {save_path}")
        else:
            plt.show()

    def plot_shap_summary(self, save_path=None):
        print("🔍 Generating SHAP summary plot...")
        explainer = shap.Explainer(self.model, self.X_train)
        shap_values = explainer(self.X_test)

        plt.figure()
        shap.summary_plot(shap_values, self.X_test, show=False)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"📁 SHAP summary plot saved to: {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train ML model for Pernicious Anaemia detection.")
    parser.add_argument("--data", type=str, required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--model", type=str, choices=["logistic", "rf", "xgb"], default="logistic", help="Model type.")
    parser.add_argument("--savefigs", action="store_true", help="Save plots to disk.")
    parser.add_argument("--output_model", type=str, default="output/model.pkl", help="Path to save trained model.")
    parser.add_argument("--output_figs_dir", type=str, default="output/figs", help="Directory to save plots.")
    args = parser.parse_args()

    trainer = PAModelTrainer(data_path=args.data, model_type=args.model)
    trainer.load_data()
    trainer.train_model()
    trainer.evaluate_model()

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(trainer.model, args.output_model)
    print(f"🧠 Model saved to: {args.output_model}")

    if args.savefigs:
        trainer.plot_feature_importance(save_path=f"{args.output_figs_dir}/feature_importance_{args.model}.png")
        trainer.plot_confusion_matrix(save_path=f"{args.output_figs_dir}/confusion_matrix_{args.model}.png")
        trainer.plot_roc_pr_curves(save_path=f"{args.output_figs_dir}/roc_pr_curve_{args.model}.png")
        trainer.plot_shap_summary(save_path=f"{args.output_figs_dir}/shap_summary_{args.model}.png")
    else:
        trainer.plot_feature_importance()
        trainer.plot_confusion_matrix()
        trainer.plot_roc_pr_curves()
        trainer.plot_shap_summary()


if __name__ == "__main__":
    main()
