import os
import argparse
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
from pa_model_trainer import PAModelTrainer


def benchmark_models(data_path, output_csv="output/model_comparison.csv"):
    results = []

    for model_type in ["logistic", "rf"]:
        print(f"\n🚀 Benchmarking model: {model_type.upper()}")
        trainer = PAModelTrainer(data_path=data_path, model_type=model_type)
        trainer.load_data()
        trainer.train_model()

        y_pred = trainer.model.predict(trainer.X_test)
        y_prob = trainer.model.predict_proba(trainer.X_test)[:, 1]

        metrics = {
            "Model": model_type.upper(),
            "AUC": roc_auc_score(trainer.y_test, y_prob),
            "Accuracy": accuracy_score(trainer.y_test, y_pred),
            "Precision": precision_score(trainer.y_test, y_pred, zero_division=0),
            "Recall": recall_score(trainer.y_test, y_pred, zero_division=0),
            "F1": f1_score(trainer.y_test, y_pred, zero_division=0)
        }
        results.append(metrics)

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
    args = parser.parse_args()

    benchmark_models(data_path=args.data, output_csv=args.output_csv)