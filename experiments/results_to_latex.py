import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import json

from utils.args import get_args

RESULTS_FILE = "results.json"
LABEL = "results"
AGGREGATORS = ["bootstrapped_converged"]
METRICS = ["transfer_area_under_curve", "adaptive_efficiency"]
COLUMNS = ["mean", "std"]


def make_parser():
    parser = argparse.ArgumentParser()

    parser.description = "Script to take the result.json file generated and print the latex tables associated with them."

    parser.add_argument("--results-file", "-rf", type=str, default=RESULTS_FILE)
    parser.add_argument("--label", "-l", type=str, default=LABEL)
    parser.add_argument("--aggregators", "-a", type=str, nargs="+", default=AGGREGATORS)
    parser.add_argument("--metrics", "-m", type=str, nargs="+", default=METRICS)
    parser.add_argument("--columns", "-cl", type=str, nargs="+", default=COLUMNS)

    return parser


def load_results(results_file):
    with open(f"./data/{results_file}", "r") as file:
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
    results, metric_names, aggregators, columns=None, label="results"
):
    def format_num(num: float) -> str:
        return f"{num:.2e}"

    latex_code = ""

    for aggregator in aggregators:
        header = "\\textbf{Metric} & \\textbf{Experiment}"
        for col in columns:
            header += f" & \\textbf{{{col.title()}}}"
        header += f" \\\\"
        rows = []

        for metric_name in metric_names:
            for i, experiment in enumerate(results):
                if i == 0:
                    row = f"\\multirow{{{len(results)}}}{{*}}{{\\textbf{{{metric_name.replace('_', ' ').title()}}}}} & {experiment.split('_')[-1].upper()}"
                else:
                    row = f" & {experiment.split('_')[-1].upper()}"
                if (
                    aggregator in results[experiment]
                    and metric_name in results[experiment][aggregator]
                ):
                    metric_data = results[experiment][aggregator][metric_name]
                    for col in columns:
                        row += f" & {format_num(metric_data[col])}"
                    row += f" \\\\"
                else:
                    row += " &" * len(columns) + f" \\\\"
                rows.append(row)
            rows.append("\\hline")

        # Generate LaTeX code
        table_code = "\\begin{table}[h]\n\t\\centering"
        table_code += f"\n\t\\begin{{tabular}}{{{'|c' * (len(columns) + 2) + '|'}}}\n\t\t\\hline\n\t\t{header}\n\t\t\\hline\n\t\t"
        table_code += "\n\t\t".join(rows)
        table_code += "\n\t\\end{tabular}"
        table_code += f"\n\t\\caption{{Metrics from each experiment ({aggregator.replace('_', ' ').title()} Aggregator).}}"
        table_code += f"\n\t\\label{{tab:{label}}}"
        table_code += "\n\\end{table}\n\n"

        table_code = table_code.replace("\t", "    ")

        latex_code += table_code

    return latex_code


def main():
    parser = make_parser()
    args = get_args(parser=parser)

    results = load_results(args.results_file)

    print(
        generate_latex_tables(
            results, args.metrics, args.aggregators, args.columns, args.label
        )
    )


if __name__ == "__main__":
    main()
