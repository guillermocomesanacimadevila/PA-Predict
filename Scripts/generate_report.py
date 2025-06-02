import pandas as pd
from jinja2 import Template
import argparse
import datetime
import os
import base64


def encode_image_base64(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def generate_html_report(csv_path, output_path, template_path, figs_dir=None):
    df = pd.read_csv(csv_path)

    images = {}
    if figs_dir:
        for name in ["confusion_matrix", "feature_importance", "roc_pr_curve", "shap_summary"]:
            for model in ["LOGISTIC", "RF", "XGB", "SVM"]:
                key = f"{name}_{model}"
                path = os.path.join(figs_dir, f"{name.lower()}_{model.lower()}.png")
                images[key] = encode_image_base64(path)

    with open(template_path, "r") as f:
        template = Template(f.read())

    rendered = template.render(
        date=str(datetime.datetime.now().date()),
        rows=df.to_dict(orient="records"),
        images=images
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(rendered)

    print(f"📄 HTML report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from model comparison results.")
    parser.add_argument("--csv", type=str, required=True, help="Path to model comparison CSV file.")
    parser.add_argument("--output", type=str, default="output/report.html", help="Output HTML report file path.")
    parser.add_argument("--template", type=str, default="Scripts/report_template.html", help="Path to Jinja2 HTML template.")
    parser.add_argument("--figs_dir", type=str, default="output/figs", help="Directory containing figure PNGs.")
    args = parser.parse_args()

    generate_html_report(args.csv, args.output, args.template, args.figs_dir)


if __name__ == "__main__":
    main()
