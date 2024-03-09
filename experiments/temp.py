import matplotlib.pyplot as plt

from paper_results import load_results

ALG_ORDER = list(
    reversed(
        [
            "None",
            "Noisy",
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
    )
)


def plot_results(results, agg, metric, title, include_axis_labels):
    labels = list(results.keys())
    height = 0.6

    plt.figure(figsize=(30, 15))
    plt.rc("font", size=40)
    ax = plt.gca()

    replace_names = {"noisy": "Noisy", "none": "None", "girm": "GIRL"}

    def alg_name(s):
        for k in replace_names:
            if k in s:
                return replace_names[k]
        return s.split("_")[-1].upper()

    alg_names_to_exp = {alg_name(exp_name): exp_name for exp_name in results}

    for k, alg in enumerate(ALG_ORDER):
        d = results[alg_names_to_exp[alg]][agg][metric]

        ax.barh(
            y=k,
            left=d[f"iq_bootstrapped_ci_95_lower"],
            width=d[f"iq_bootstrapped_ci_95_upper"] - d[f"iq_bootstrapped_ci_95_lower"],
            alpha=0.7,
            label=alg,
            height=height,
        )
        ax.vlines(
            x=d[f"iq_mean"],
            ymin=k - height / 2,
            ymax=k + height / 2,
            label=alg,
            color="k",
            alpha=1,
            linewidth=5,
        )
    if include_axis_labels:
        ax.set_yticks(list(range(len(labels))))
        ax.set_yticklabels(ALG_ORDER)
    else:
        ax.set_yticks([])
    # plt.boxplot(plot_data, labels=labels, vert=False, showfliers=False)
    plt.title(title)
    plt.savefig(f"./figures/{title.replace(' ', '_').lower()}.png")
    plt.close()


if __name__ == "__main__":
    door_key_results = load_results("results/door_key_change/data/results.json")
    results = plot_results(
        door_key_results,
        "converged_all",
        "adaptive_efficiency",
        "IQM of Adaptive Efficiency on DoorKeyChange",
        True,
    )
    results = plot_results(
        door_key_results,
        "converged_all",
        "transfer_area_under_curve",
        "IQM of TrAUC on DoorKeyChange",
        False,
    )
    walker_results = load_results("results/walker_thigh_length/data/results.json")
    results = plot_results(
        walker_results,
        "converged_all",
        "adaptive_efficiency",
        "IQM of Adaptive Efficiency on WalkerThighIncrease",
        False,
    )
    results = plot_results(
        walker_results,
        "converged_all",
        "transfer_area_under_curve",
        "IQM of TrAUC on WalkerThighIncrease",
        False,
    )
