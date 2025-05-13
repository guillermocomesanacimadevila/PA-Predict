import pandas as pd
from jinja2 import Template
import datetime
import argparse
import os

def generate_html_report(csv_path, output_path="output/report.html", template_path="report_template.html"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"❌ Template file not found: {template_path}")

    df = pd.read_csv(csv_path)

    with open(template_path, "r") as f:
        template = Template(f.read())

    rendered = template.render(
        date=str(datetime.datetime.now().date()),
        rows=df.to_dict(orient="records")
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(rendered)

    print(f"📄 Report generated: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML report from model benchmark CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file (e.g., model_comparison.csv).")
    parser.add_argument("--output", type=str, default="output/report.html", help="Path to output HTML report.")
    parser.add_argument("--template", type=str, default="report_template.html", help="Path to HTML Jinja2 template.")
    args = parser.parse_args()

    generate_html_report(args.csv, args.output, args.template)
