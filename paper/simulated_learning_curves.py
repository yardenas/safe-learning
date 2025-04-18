# %%
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn.objects as so
import wandb
from seaborn import axes_style
from tueplots import bundles, figsizes

warnings.filterwarnings("ignore")


# %%
api = wandb.Api()


def load_evaluation(data, seed, category, environment):
    if data.index.name is None or data.index.name != "step":
        data.rename(columns={"_step": "step"}, inplace=True)
        data.set_index("step", inplace=True)
    data = data[data.columns[data.columns.str.startswith("eval")]].dropna()
    data["seed"] = seed
    data["category"] = category
    data["environment"] = environment
    return data


def config_to_category(config):
    if "cost_robustness" in config["agent"]:
        name = config["agent"]["cost_robustness"]["name"]
        if name == "ramu":
            return "ramu"
        return "ptsd"
    elif config["training"]["train_domain_randomization"]:
        return "dr"
    elif config["training"]["eval_domain_randomization"]:
        return "nominal"
    else:
        return "simple"


def handle_run(run):
    metrics = run.history(
        keys=["eval/episode_reward", "eval/episode_cost", "_step"],
        x_axis="_step",
        pandas=False,
    )
    safe = run.config["training"]["safe"]
    if not safe:
        return
    config = run.config
    environment = config["environment"]["task_name"]
    metrics = pd.DataFrame(metrics)
    if environment == "rccar":
        environment = "RaceCar"
    if environment == "go_to_goal":
        environment = "PointGoal2"
    category = config_to_category(config)
    seed = config["training"]["seed"]
    return metrics, seed, category, environment


def walk_wandb_runs(project, filters):
    runs = api.runs(project, filters=filters)
    print(f"Fetching {len(runs)} runs")
    for run in runs:
        out = handle_run(run)
        if out is None:
            continue
        yield out


filters = {
    "display_name": {
        "$regex": "apr01-nominal-aga$|apr01-dr-aga$|apr01-ptsd-aga|mar28-simple-aga|apr01-ramu-aga|apr11-g2g-ramu|apr11-g2g-ptsd|apr11-g2g-dr|apr11-g2g-nominal|apr11-g2g-simple-aga$"
    }
}
data = pd.concat(
    [
        load_evaluation(*summary_files_data)
        for summary_files_data in walk_wandb_runs("yardas/ss2r", filters)
    ]
)
# %%
budgets = {
    "RaceCar": 5,
    "SafeCartpoleSwingup": 100,
    "SafeHumanoidWalk": 100,
    "SafeQuadrupedRun": 100,
    "SafeWalkerWalk": 100,
    "PointGoal2": 25,
}

reference_values = (
    data[data["category"] == "simple"]
    .set_index("environment")["eval/episode_reward"]
    .to_dict()
)
# Normalizing the eval/episode_reward for each row based on the corresponding "simple" value
data["normalized_episode_reward"] = data.apply(
    lambda row: row["eval/episode_reward"] / reference_values[row["environment"]],
    axis=1,
)

data["normalized_episode_cost"] = data.apply(
    lambda row: row["eval/episode_cost"] / budgets[row["environment"]], axis=1
)
# %%


metrics = ["eval/episode_reward", "eval/episode_cost"]
y_labels = {
    metrics[0]: r"$\hat{J}_r$",
    metrics[1]: r"$\hat{J}_c$",
}


def create_env_plot(env_names, share_y=True, share_x=True):
    theme = bundles.neurips2024()
    so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
    plt.rcParams.update(bundles.neurips2024())
    plt.rcParams.update(figsizes.neurips2024(nrows=2, ncols=len(env_names)))
    fig = plt.figure()
    marker_styles = ["o", "x", "^", "s", "*"]
    plot_data = data[data["environment"].isin(env_names)]
    so.Plot(plot_data, x="step", color="category", marker="category").pair(
        y=metrics
    ).facet(col="environment", order=env_names).share(x=share_x, y=share_y).add(
        so.Line(linewidth=1.0, pointsize=2.5, edgewidth=0.1),
        so.Agg("median"),
    ).add(
        so.Band(alpha=0.15), so.Est("median", errorbar=("ci", 68)), legend=False
    ).scale(
        color=so.Nominal(
            values=[
                "#5F4690",
                "#1D6996",
                "#38A6A5",
                "#0F8554",
                "#73AF48",
                "#EDAD08",
                "#E17C05",
                "#CC503E",
                "#94346E",
                "#6F4070",
                "#994E95",
                "#666666",
            ],
            order=["ptsd", "ramu", "dr", "nominal"],
        ),
        marker=so.Nominal(values=marker_styles),
    ).label(
        x="",
        y=lambda name: y_labels[name],
        title=lambda x: x.strip("Safe"),
    ).theme(axes_style("ticks")).on(fig).plot()
    for ax in fig.get_axes():
        ax.grid(True, linewidth=0.5, c="gainsboro", axis="both", zorder=0)
    legend = fig.legends.pop(0)
    text = {
        "ramu": "\sf RAMU",
        "ptsd": "\sf PTSD",
        "dr": "\sf Domain Randomization",
        "nominal": "\sf Nominal",
    }
    handles = []
    for handle, marker in zip(legend.legend_handles, marker_styles):
        handle.set_marker(marker)
        handle.set_linewidth(0)
        handle.set_markersize(3.5)
        handles.append(handle)
    fig.legend(
        handles,
        [text[t.get_text()] for t in legend.texts],
        loc="center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=6,
        frameon=False,
        handletextpad=0.25,
        handlelength=1.0,
    )
    return fig


fig = create_env_plot(["PointGoal2"])
fig.savefig("simulated-curves-safety-gym.pdf")
rwrl_envs = [
    "SafeCartpoleSwingup",
    "SafeHumanoidWalk",
    "SafeQuadrupedRun",
    "SafeWalkerWalk",
]
fig = create_env_plot(rwrl_envs, share_x=False, share_y=False)

count = 0
for ax in fig.get_axes():
    ax.grid(True, linewidth=0.5, c="gainsboro", axis="both", zorder=0)
    if ax.get_subplotspec().is_last_row():
        budget = budgets[rwrl_envs[count]]
        ax.axhline(
            y=budgets[rwrl_envs[count]],
            color="black",
            linestyle=(0, (1, 1)),
            linewidth=1.25,
        )
        yticks = ax.get_yticks()
        y_new = budget
        ax.set_yticks(list(yticks) + [y_new])
        ytick_labels = [f"${tick:.0f}$" for tick in yticks] + ["Budget"]
        ax.set_yticklabels(ytick_labels)
        count += 1

fig.savefig("simulated-curves-rwrl.pdf")
# %%
env_names = data.environment.unique()
n_envs = len(env_names)
n_metrics = len(metrics)
n_total = n_envs * n_metrics
ncols = 3
nrows = 4
theme = bundles.neurips2024()
so.Plot.config.theme.update(axes_style("white") | theme | {"legend.frameon": False})
plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update(
    figsizes.neurips2024(nrows=nrows, ncols=ncols, height_to_width_ratio=0.75)
)
fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=plt.rcParams["figure.figsize"],
    sharex=False,
    sharey=False,
)
plot_data = data[data["environment"].isin(env_names)]
marker_styles = ["o", "x", "^", "s", "*"]
color_order = ["ptsd", "ramu", "dr", "nominal"]
colors = [
    "#5F4690",
    "#1D6996",
    "#38A6A5",
    "#0F8554",
    "#73AF48",
    "#EDAD08",
    "#E17C05",
    "#CC503E",
    "#94346E",
    "#6F4070",
    "#994E95",
    "#666666",
]
text = {
    "ramu": "\sf RAMU",
    "ptsd": "\sf PTSD",
    "dr": "\sf Domain Randomization",
    "nominal": "\sf Nominal",
}
for idx, env in enumerate(env_names):
    col = idx % ncols
    row_base = 2 * (idx // ncols)  # two rows per environment
    for i, metric in enumerate(metrics):
        row = row_base + i
        ax = axes[row, col]
        env_data = plot_data[plot_data["environment"] == env]
        show_ylabel = col == 0  # Only on leftmost column
        p = (
            so.Plot(env_data, x="step", y=metric, color="category", marker="category")
            .add(
                so.Line(linewidth=1.0, pointsize=2.5, edgewidth=0.1),
                so.Agg("median"),
            )
            .add(
                so.Band(alpha=0.15), so.Est("median", errorbar=("ci", 68)), legend=False
            )
            .scale(
                color=so.Nominal(values=colors, order=color_order),
                marker=so.Nominal(values=marker_styles),
            )
            .label(
                x="",
                y=y_labels[metric] if show_ylabel else "",
                title=env.strip("Safe"),
            )
            .theme(axes_style("ticks"))
            .on(ax)
            .plot()
        )
        ax.grid(True, linewidth=0.5, c="gainsboro", axis="both", zorder=0)
        if row in [1, 3]:
            budget = budgets[env]
            ax.axhline(y=budget, color="black", linestyle=(0, (1, 1)), linewidth=1.25)
            ylim = ax.get_ylim()
            yticks = ax.get_yticks()
            y_new = budget
            ax.set_yticks(list(yticks) + [y_new])
            ytick_labels = [f"${tick:.0f}$" for tick in yticks] + ["Budget"]
            ax.set_yticklabels(ytick_labels)
            ax.set_ylim(*ylim)

legend = fig.legends.pop(0)
fig.legends = []
fig.legend(
    legend.legend_handles,
    [text[t.get_text()] for t in legend.texts],
    loc="center",
    bbox_to_anchor=(0.5, 1.015),
    ncol=4,
    frameon=False,
)
fig.supxlabel("Simulation Steps", fontsize=9)
fig.savefig("simulated-curves.pdf")
