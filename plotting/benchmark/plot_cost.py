from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "axes.unicode_minus": False,
        "text.latex.preamble": (
            r"\usepackage{amsmath} "
            r"\usepackage{amsfonts} "
            r"\usepackage{amssymb} "
            r"\usepackage{bm} "
        ),
    }
)

RESULTS_DIR = Path("benches/results")
FIGURES_DIR = Path("figures/benchmark")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["Time (s)", "Memory Estimate (GiB)", "Allocations"]
METRIC_TITLES = ["Time", "Memory", "Allocations"]

PALETTE = {
    "KAEM": "#2ecc71",
    "VAE": "#3498db",
    "Pang": "#e67e22",
}

COL_RENAME = {
    "time_mean": "Time (s)",
    "time_std": "Time Std (s)",
    "memory_estimate": "Memory Estimate (GiB)",
    "allocations": "Allocations",
}


def load_csv_safe(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    print(f"Warning: {path} not found, skipping.")
    return None


def normalize(df: pd.DataFrame, key: str, model: str, q_from_n_z: bool):
    out = df.rename(columns=COL_RENAME).copy()
    out["Model"] = model
    out["Latent Dim"] = (2 * df[key] + 1) if q_from_n_z else df[key]
    return out


def plot_grouped_bars(
    entries: list[tuple[pd.DataFrame, str]],
    title: str,
    output_name: str,
):
    """Plot grouped bars across models. entries: list of (dataframe, label)."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))

    latent_dims = sorted(entries[0][0]["Latent Dim"].unique())
    n_models = len(entries)
    width = 0.8 / n_models
    x = np.arange(len(latent_dims))

    for idx, (metric, metric_title) in enumerate(zip(METRICS, METRIC_TITLES)):
        ax = axs[idx]

        for m_idx, (df, label) in enumerate(entries):
            offset = (m_idx - (n_models - 1) / 2) * width
            values = [
                df[df["Latent Dim"] == ld][metric].values[0] for ld in latent_dims
            ]
            kwargs = {
                "label": label,
                "color": PALETTE[label],
                "edgecolor": "white",
                "linewidth": 0.5,
            }

            if metric == "Time (s)" and "Time Std (s)" in df.columns:
                stds = [
                    df[df["Latent Dim"] == ld]["Time Std (s)"].values[0]
                    for ld in latent_dims
                ]
                ax.bar(
                    x + offset,
                    values,
                    width,
                    yerr=stds,
                    capsize=3,
                    error_kw={"elinewidth": 1.2, "capthick": 1.2, "alpha": 0.8},
                    **kwargs,
                )
            else:
                ax.bar(x + offset, values, width, **kwargs)

        ax.set_xticks(x)
        ax.set_xticklabels(latent_dims, fontsize=14)
        ax.set_xlabel(r"Latent Dim, $Q$", fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.set_title(metric_title, fontsize=17, pad=8)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=n_models,
        fontsize=15,
        frameon=False,
    )
    fig.suptitle(title, fontsize=19, y=1.08)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_only(
    entries: list[tuple[pd.DataFrame, str]],
    title: str,
    output_name: str,
):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    latent_dims = sorted(entries[0][0]["Latent Dim"].unique())
    n_models = len(entries)
    width = 0.8 / n_models
    x = np.arange(len(latent_dims))

    for m_idx, (df, label) in enumerate(entries):
        offset = (m_idx - (n_models - 1) / 2) * width
        times = [df[df["Latent Dim"] == ld]["Time (s)"].values[0] for ld in latent_dims]
        stds = [
            df[df["Latent Dim"] == ld]["Time Std (s)"].values[0] for ld in latent_dims
        ]
        ax.bar(
            x + offset,
            times,
            width,
            yerr=stds,
            label=label,
            color=PALETTE[label],
            capsize=4,
            error_kw={"elinewidth": 1.5, "capthick": 1.5, "alpha": 0.8},
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(latent_dims, fontsize=15)
    ax.set_xlabel(r"Latent Dim, $Q$", fontsize=18)
    ax.set_ylabel("Time (s)", fontsize=18)
    ax.set_title(title, fontsize=19, pad=10)
    ax.legend(fontsize=14, frameon=False, loc="upper left")
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_temperatures(df: pd.DataFrame, output_name: str):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    color = sns.color_palette("viridis", n_colors=4)[1]

    for idx, (metric, metric_title) in enumerate(zip(METRICS, METRIC_TITLES)):
        ax = axs[idx]
        col_map = {
            "Time (s)": "time_mean",
            "Memory Estimate (GiB)": "memory_estimate",
            "Allocations": "allocations",
        }
        values = df[col_map[metric]].values
        x = np.arange(len(df))

        if metric == "Time (s)":
            ax.bar(
                x,
                values,
                yerr=df["time_std"].values,
                color=color,
                capsize=4,
                error_kw={"elinewidth": 1.5, "capthick": 1.5, "alpha": 0.8},
                edgecolor="white",
                linewidth=0.5,
            )
        else:
            ax.bar(x, values, color=color, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(df["N_t"].values, fontsize=14)
        ax.set_xlabel(r"$N_t$", fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.set_title(metric_title, fontsize=17, pad=8)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Power Posteriors", fontsize=19, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    kaem_train = load_csv_safe(RESULTS_DIR / "latent_dim.csv")
    kaem_sample = load_csv_safe(RESULTS_DIR / "ITS_generation.csv")
    temperatures = load_csv_safe(RESULTS_DIR / "temperatures.csv")

    vae_train = load_csv_safe(RESULTS_DIR / "vae_latent_dim.csv")
    vae_sample = load_csv_safe(RESULTS_DIR / "vae_sampling.csv")

    pang_train = load_csv_safe(RESULTS_DIR / "pang_latent_dim.csv")
    pang_sample = load_csv_safe(RESULTS_DIR / "pang_sampling.csv")

    # Training comparison
    train_entries = []
    if kaem_train is not None:
        train_entries.append((normalize(kaem_train, "n_z", "KAEM", True), "KAEM"))
    if vae_train is not None:
        train_entries.append((normalize(vae_train, "latent_dim", "VAE", False), "VAE"))
    if pang_train is not None:
        train_entries.append(
            (normalize(pang_train, "latent_dim", "Pang", False), "Pang")
        )

    if len(train_entries) >= 2:
        plot_grouped_bars(
            train_entries,
            "Training Cost",
            "01_training_comparison.png",
        )
        print("Saved: 01_training_comparison.png")

        plot_time_only(
            train_entries,
            "Training Time",
            "01b_training_time_only.png",
        )
        print("Saved: 01b_training_time_only.png")

    # Sampling comparison
    sample_entries = []
    if kaem_sample is not None:
        sample_entries.append((normalize(kaem_sample, "n_z", "KAEM", True), "KAEM"))
    if vae_sample is not None:
        sample_entries.append(
            (normalize(vae_sample, "latent_dim", "VAE", False), "VAE")
        )
    if pang_sample is not None:
        sample_entries.append(
            (normalize(pang_sample, "latent_dim", "Pang", False), "Pang")
        )

    if len(sample_entries) >= 2:
        plot_grouped_bars(
            sample_entries,
            "Sampling Cost",
            "02_sampling_comparison.png",
        )
        print("Saved: 02_sampling_comparison.png")

        plot_time_only(
            sample_entries,
            "Sampling Time",
            "02b_sampling_time_only.png",
        )
        print("Saved: 02b_sampling_time_only.png")

    # Power posteriors
    if temperatures is not None:
        plot_temperatures(temperatures, "03_training_vs_num_temperatures.png")
        print("Saved: 03_training_vs_num_temperatures.png")


if __name__ == "__main__":
    main()
