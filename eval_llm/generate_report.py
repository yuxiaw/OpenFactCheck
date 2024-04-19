import sys

from pathlib import Path
import os
import subprocess
import argparse
from jinja2 import Environment, FileSystemLoader


def create_latex_from_template(args, destination_path):
    """
    Fill data in tex templates.
    """

    loader = FileSystemLoader(args.tex_template_path)
    env = Environment(loader=loader)
    data = {
        "model_name": args.model_name.replace("_", " "),
        # "free_style_response_barchart_path": "free_style_response_barchart.pdf",
        # "selfaware_cm_path": "selfaware_cm.pdf",
        # "selfaware_performance_path": "selfaware_performance.pdf",
        # "freshqa_piechart_path": "freshqa_piechart.pdf",
        # "snowballing_acc_path": "snowballing_acc.pdf",
    }
    template = env.get_template(args.tex_template_name)
    latex = template.render(data)
    with open(Path(destination_path) / ("main.tex"), "w", encoding="utf-8") as f:
        f.write(latex)

    return None


def compile_pdf(args, destination_path):
    """
    Compile the latex file to pdf.
    """

    base_path = os.path.dirname(os.path.abspath(__file__))
    latex_file_dir = os.path.join(base_path, destination_path)
    os.chdir(latex_file_dir)
    try:
        print("First Compilation")
        subprocess.run(["pdflatex", "main.tex"])
        print("Second Compilation")
        subprocess.run(["pdflatex", "main.tex"])
        print(
            "The report has been generated\nmain.pdf"
        )
    except Exception as e:
        print(f"An error occurred: {e}")


def create_pdf(args):
    """
    Create a pdf report.
    """

    latex_path = Path(args.input_path)
    create_latex_from_template(args, latex_path)
    compile_pdf(args, latex_path)


def main(args):
    create_pdf(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="Moonshot",
        # default="Baichuan",
        help="The model to be evaluated",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./figure",
        # default="../dev/backup/Baichuan.xlsx",
        help="Path to the response data",
    )
    parser.add_argument(
        "--tex_template_path",
        type=str,
        default="latex_template",
        help="The path to the tex template",
    )
    parser.add_argument(
        "--tex_template_name",
        type=str,
        default="fact_checking_template.tex",
        help="The default template name",
    )

    args = parser.parse_args()
    main(args)
