import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import json
import numpy as np

from utils.args import get_args
from utils.arg_types import str2bool

RESULTS_FILE = "./data/results.json"
LABEL = "results"
AGGREGATORS = [
    "converged_0",
    "converged_all",
    "all",
    "converged_0",
    "all",
    "converged_0",
]
METRICS = [
    "convergence_efficiency",
    "adaptive_efficiency",
    "transfer_area_under_curve",
    "transfer_area_under_curve",
    "convergence_freq",
    "adaptive_freq",
]  # could add retention and resiliance
LOWER_BETTER = [
    True,
    True,
    False,
    False,
    False,
    False,
]
CAPTIONS = [
    "This table shows the convergence efficiency on the pre-novelty task. It is computed by calculating the number of steps from the start of training until convergence on the first task. Thus, lower numbers are better here. Only runs that converged on the first task are taken into account for this metric.",
    "This table shows the adaptive efficiency on the post-novelty task. it is computed by calculating the number of steps from the start of the novel task until convergence on the second task. Thus, lower numbers are better. Only runs that converged on both tasks are taken into account for this metric.",
    "This table shows the transfer area under the curve metric, which is computed by adding final reward on the first task with the area under the reward curve in the second task. Higher numbers are better here. This includes all the runs.",
    "This table shows the transfer area under the curve metric, which is computed by adding final reward on the first task with the area under the reward curve in the second task. Higher numbers are better here. This only includes runs that converged on the first task.",
    "This is the frequency that the agent converges on the first task using this exploration algorithm. Higher numbers are better.",
    "This is the frequency that the agent converges on the second task using this exploration algorithm conditioned on the fast it converged on the first task. Higher numbers are better.",
]
PREFIX = ""

SORT_ROWS = [
    "None (PPO)",
    "NoisyNets",
    "ICM",
    "DIAYN",
    "RND",
    "NGU",
    "RIDE",
    "GIRL",
    "RE3",
    "RISE",
    "REVD",
]

RESULTS_DIRS = [
    "door_key_change",
    "lava_maze_safe_to_hurt",
    "lava_maze_hurt_to_safe",
    "simple_to_lava_crossing",
    "walker_thigh_length",
]
COLUMN_NAMES = [
    "DoorKeyChange",
    "LavaNotSafe",
    "LavaProof",
    "CrossingBarrier",
    "ThighIncrease",
]


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
    results,
    env_names,
    metrics_names,
    aggregators,
    lower_better,
    captions,
    prefix,
    label,
):
    def calc_exponent(value):
        exp = 0
        while value < 1:
            exp -= 1
            value *= 10
        while value > 10:
            exp += 1
            value /= 10
        return value, exp

    def force_exponent(value, exp):
        return value / (10**exp)

    def format_num(mean, std, bold, exp):
        mean_val = float(f"{force_exponent(mean, exp):.3g}")
        std_val = float(f"{force_exponent(std, exp):.3g}")

        s = f"{mean_val} $\\pm$ {std_val}"
        if bold:
            s = f"\\textbf{{{s}}}"
        return s

    assert len(metrics_names) == len(aggregators)

    n_novelties = len(env_names)

    latex_code = ""

    replace_names = {"noisy": "NoisyNets", "none": "None (PPO)", "girm": "GIRL"}

    def alg_name(s):
        for k in replace_names:
            if k in s:
                return replace_names[k]
        return s.split("_")[-1].upper()

    for metric, agg, down, caption in zip(
        metrics_names, aggregators, lower_better, captions
    ):
        data = {}
        exponent_per_env = {}
        for name in env_names:
            for exp_id in results[name]:
                a_name = alg_name(exp_id)
                if a_name not in data:
                    data[a_name] = {}
                data[a_name][name] = [
                    results[name][exp_id][agg][metric][f"{prefix}mean"],
                    results[name][exp_id][agg][metric][f"{prefix}std"],
                    False,
                ]
            algs = [alg for alg in data]
            exponent_per_env[name] = max(
                [calc_exponent(data[alg][name][0])[1] for alg in data]
            )
            if down:
                m = np.min([data[alg][name][0] for alg in data])
            else:
                m = np.max([data[alg][name][0] for alg in data])
            for alg in algs:
                if np.isclose(m, data[alg][name][0]):
                    data[alg][name][2] = True

        header = f" & \\multicolumn{{{n_novelties}}}{{|c|}}{{{metric.replace('_', ' ').title()} \\{'down' if down else 'up'}arrow}} \\\\ \\cline{{2-{n_novelties + 1}}} \n\t\tExploration"
        for name in COLUMN_NAMES:
            header += f" & {name.replace('_', ' ')}"
        header += f" \\\\ \n\t\tAlgorithm"
        for name in env_names:
            header += f" & (10^{{{exponent_per_env[name]}}})"
        header += f" \\\\ \\hline \n"

        rows = []

        for alg in data:
            row = alg
            for name in env_names:
                val = data[alg][name]
                row += f" & {format_num(*val, exponent_per_env[name])}"
            rows.append(row + f" \\\\")

        new_rows = []

        for alg in SORT_ROWS:
            found = False
            for row in rows:
                if row.lower().startswith(alg.lower()):
                    new_rows.append(row)
                    found = True
                    continue
            if not found:
                raise ValueError(f"Algorithm {alg.upper()} not found!")

        rows = new_rows
        rows.insert(1, "\\hline")

        rows.append("\\hline")

        if caption is None:
            caption = "Data for {metric} using the {agg} aggregator."

        # Generate LaTeX code
        table_code = "\\begin{table}[h]\n\t\\centering"
        table_code += f"\n\t\\begin{{tabular}}{{{'|c' * (n_novelties + 1) + '|'}}}\n\t\t\\hline\n\t\t{header}\n\t\t\\hline\n\t\t"
        table_code += "\n\t\t".join(rows)
        table_code += "\n\t\\end{tabular}"
        table_code += f"\n\t\\caption{{{caption}}}"
        table_code += f"\n\t\\label{{tab:{label}}}"
        table_code += "\n\\end{table}\n\n"

        table_code = table_code.replace("\t", "    ")

        latex_code += table_code

    return latex_code


def main():
    results = {
        d.replace("tuned_", ""): load_results(f"./results/{d}/{RESULTS_FILE}")
        for d in RESULTS_DIRS
    }
    result_names = [d.replace("tuned_", "") for d in RESULTS_DIRS]

    print(
        generate_latex_tables(
            results,
            result_names,
            METRICS,
            AGGREGATORS,
            LOWER_BETTER,
            CAPTIONS,
            PREFIX,
            LABEL,
        )
    )


if __name__ == "__main__":
    main()
