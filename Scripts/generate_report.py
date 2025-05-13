import pandas as pd
from jinja2 import Template
import argparse
import datetime
import os


def generate_html_report(csv_path, output_path, template_path):
    # Load model comparison CSV
    df = pd.read_csv(csv_path)

    # Load HTML template
    with open(template_path, "r") as f:
        template = Template(f.read())

    # Render the HTML report
    rendered = template.render(
        date=str(datetime.datetime.now().date()),
        rows=df.to_dict(orient="records")
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the report
    with open(output_path, "w") as f:
        f.write(rendered)

    print(f"📄 HTML report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from model comparison results.")
    parser.add_argument("--csv", type=str, required=True, help="Path to model comparison CSV file.")
    parser.add_argument("--output", type=str, default="output/report.html", help="Output HTML report file path.")
    parser.add_argument("--template", type=str, default="Scripts/report_template.html", help="Path to Jinja2 HTML template.")

    args = parser.parse_args()

    generate_html_report(csv_path=args.csv, output_path=args.output, template_path=args.template)


if __name__ == "__main__":
    main()
