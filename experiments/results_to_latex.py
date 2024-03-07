import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import json

from utils.args import get_args
from utils.arg_types import str2bool

RESULTS_FILE = "./data/results.json"
LABEL = "results"
AGGREGATORS = ["bootstrapped_converged", "converged", "all"]
METRICS = ["transfer_area_under_curve", "adaptive_efficiency"]
PREFIX_LIST = ["", "iq_"]
ROTATE = True


def make_parser():
    parser = argparse.ArgumentParser()

    parser.description = "Script to take the result.json file generated and print the latex tables associated with them."

    parser.add_argument("--results-file", "-rf", type=str, default=RESULTS_FILE)
    parser.add_argument("--label", "-l", type=str, default=LABEL)
    parser.add_argument("--aggregators", "-a", type=str, nargs="+", default=AGGREGATORS)
    parser.add_argument("--metrics", "-m", type=str, nargs="+", default=METRICS)
    parser.add_argument(
        "--prefix-list", "-pl", type=str, nargs="+", default=PREFIX_LIST
    )
    parser.add_argument("--rotate", "-r", type=str2bool, default=ROTATE)

    return parser


def load_results(results_file):
    with open(results_file, "r") as file:
        results_data = json.load(file)
    return results_data


def get_all_metric_names(results):
    metric_names = set()

    for result in results.values():
        for metric_dict in result.values():
            if isinstance(metric_dict, dict):
                metric_names.update(metric_dict.keys())

    return list(metric_names)


def generate_latex_tables(
    title, results, metric_names, aggregators, prefix_lst, label="results", rotate=True
):
    def format_num(num: float) -> str:
        return f"{num:.2e}"

    latex_code = ""

    for aggregator in aggregators:
        header = ""
        for metric in metric_names:
            header += f" & \\multicolumn{{{len(prefix_lst)}}}{{c|}}{{\\textbf{{{metric.replace('_', ' ').title()}}}}}"
        header += f" \\\\ \n"
        header += "\t\t\\textbf{Exploration Algorithm} & " + " & ".join(
            [
                f"{prefix.replace('_', ' ')}mean $\\pm$ {prefix.replace('_', ' ')}std"
                for _ in metric_names
                for prefix in prefix_lst
            ]
        )
        header += f" \\\\"
        rows = []

        for experiment in results:
            row = f"{experiment.split('_')[-1].upper()} & "
            for metric_name in metric_names:
                metric_data = results[experiment][aggregator][metric_name]
                for prefix in prefix_lst:
                    mu = metric_data[f"{prefix}mean"]
                    std = metric_data[f"{prefix}std"]
                    row += f"{format_num(mu)} $\\pm$ {format_num(std)} & "
            rows.append(row[:-2] + f"\\\\")
        rows.append("\\hline")

        # Generate LaTeX code
        table_code = "\\begin{table}[h]\n\t\\centering"
        table_code += f"\n\t\\begin{{tabular}}{{{'|c' * (len(metric_names) * len(prefix_lst) + 1) + '|'}}}\n\t\t\\hline\n\t\t{header}\n\t\t\\hline\n\t\t"
        table_code += "\n\t\t".join(rows)
        table_code += "\n\t\\end{tabular}"
        table_code += f"\n\t\\caption{{Metrics from each experiment ({aggregator.replace('_', ' ').title()} Aggregator). From {title.replace('_', ' ')}}}"
        table_code += f"\n\t\\label{{tab:{label}}}"
        table_code += "\n\\end{table}\n\n"

        table_code = table_code.replace("\t", "    ")

        latex_code += table_code

    latex_code = f"{title}\n{latex_code}"

    if rotate:
        latex_code = (
            f"\\begin{{landscape}}\n\t"
            + latex_code.replace("\n", "\n\t")
            + f"\n\\end{{landscape}}"
        )

    return latex_code


def main():
    parser = make_parser()
    args = get_args(parser=parser)

    results = load_results(args.results_file)

    print(
        generate_latex_tables(
            args.results_file,
            results,
            args.metrics,
            args.aggregators,
            args.prefix_list,
            args.label,
            args.rotate,
        )
    )


if __name__ == "__main__":
    main()
