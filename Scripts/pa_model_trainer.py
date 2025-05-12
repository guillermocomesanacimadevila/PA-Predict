import pandas as pd
import numpy as np
import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


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
            self.model = LogisticRegression(max_iter=1000)
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Model type must be 'logistic' or 'rf'.")
        self.model.fit(self.X_train, self.y_train)
        print(f"✅ Trained {self.model_type.upper()} model.")

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        print(f"\n📈 Evaluation for {self.model_type.upper()}:\n")
        print(classification_report(self.y_test, y_pred))
        print(f"AUC-ROC: {roc_auc_score(self.y_test, y_prob):.3f}")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.3f}")
        print(f"Precision: {precision_score(self.y_test, y_pred):.3f}")
        print(f"Recall: {recall_score(self.y_test, y_pred):.3f}")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))

    def plot_feature_importance(self, save_path=None):
        if self.model_type == 'logistic':
            importances = self.model.coef_[0]
            title = "Feature Importance (Logistic Regression)"
            xlabel = "Coefficient"
        elif self.model_type == 'rf':
            importances = self.model.feature_importances_
            title = "Feature Importance (Random Forest)"
            xlabel = "Importance"
        else:
            raise ValueError("Model type must be 'logistic' or 'rf'.")

        sorted_idx = np.argsort(np.abs(importances))[::-1]
        sorted_features = np.array(self.feature_names)[sorted_idx]
        sorted_importances = np.array(importances)[sorted_idx]

        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid", font_scale=1.2)
        sns.barplot(
            x=sorted_importances,
            y=sorted_features,
            palette="Blues_d",
            edgecolor="black"
        )
        plt.title(title, fontsize=16, weight='bold')
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel("Features", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📁 Feature importance plot saved to: {save_path}")
        else:
            plt.show()

    def plot_confusion_matrix(self, normalize=False, save_path=None):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        labels = ["No PA", "Diagnosed PA"]

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
            title = "Normalized Confusion Matrix"
        else:
            fmt = "d"
            title = "Confusion Matrix"

        plt.figure(figsize=(6, 5))
        sns.set(style="white", font_scale=1.2)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.5,
            cbar=False
        )
        plt.ylabel("Actual", fontsize=14)
        plt.xlabel("Predicted", fontsize=14)
        plt.title(title, fontsize=16, weight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📁 Confusion matrix saved to: {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train ML model for Pernicious Anaemia detection.")
    parser.add_argument("--data", type=str, required=True, help="Path to the input CSV dataset.")
    parser.add_argument("--model", type=str, choices=["logistic", "rf"], default="logistic", help="Model type.")
    parser.add_argument("--savefigs", action="store_true", help="Save feature importance and confusion matrix plots.")
    args = parser.parse_args()

    trainer = PAModelTrainer(data_path=args.data, model_type=args.model)
    trainer.load_data()
    trainer.train_model()
    trainer.evaluate_model()

    if args.savefigs:
        trainer.plot_feature_importance(save_path=f"figs/feature_importance_{args.model}.png")
        trainer.plot_confusion_matrix(normalize=True, save_path=f"figs/confusion_matrix_{args.model}.png")
    else:
        trainer.plot_feature_importance()
        trainer.plot_confusion_matrix()


if __name__ == "__main__":
    main()