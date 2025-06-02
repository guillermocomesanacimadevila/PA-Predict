import os
import argparse
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
from pa_model_trainer import PAModelTrainer


def benchmark_models(data_path, output_csv="output/model_comparison.csv", figs_dir="output/figs"):
    os.makedirs(figs_dir, exist_ok=True)
    results = []

    for model_type in ["logistic", "rf", "xgb", "svm"]:
        print(f"\n🚀 Benchmarking model: {model_type.upper()}")
        trainer = PAModelTrainer(data_path=data_path, model_type=model_type)
        trainer.load_data()
        trainer.train_model()

        y_pred = trainer.model.predict(trainer.X_test)
        y_prob = trainer.model.predict_proba(trainer.X_test)[:, 1]

        # Save metrics
        metrics = {
            "Model": model_type.upper(),
            "AUC": roc_auc_score(trainer.y_test, y_prob),
            "Accuracy": accuracy_score(trainer.y_test, y_pred),
            "Precision": precision_score(trainer.y_test, y_pred, zero_division=0),
            "Recall": recall_score(trainer.y_test, y_pred, zero_division=0),
            "F1": f1_score(trainer.y_test, y_pred, zero_division=0)
        }
        results.append(metrics)

        # Save visualizations per model
        model_tag = model_type.lower()
        trainer.plot_feature_importance(save_path=f"{figs_dir}/feature_importance_{model_tag}.png")
        trainer.plot_confusion_matrix(save_path=f"{figs_dir}/confusion_matrix_{model_tag}.png")
        trainer.plot_roc_pr_curves(save_path=f"{figs_dir}/roc_pr_curve_{model_tag}.png")
        trainer.plot_shap_summary(save_path=f"{figs_dir}/shap_summary_{model_tag}.png")

    # Save results table
    df = pd.DataFrame(results)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"\n📊 Model comparison saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ML models for PA detection.")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset CSV.")
    parser.add_argument("--output_csv", type=str, default="output/model_comparison.csv", help="Path to save comparison table.")
    parser.add_argument("--figs_dir", type=str, default="output/figs", help="Directory to save model visualizations.")
    args = parser.parse_args()

    benchmark_models(
        data_path=args.data,
        output_csv=args.output_csv,
        figs_dir=args.figs_dir
    )
