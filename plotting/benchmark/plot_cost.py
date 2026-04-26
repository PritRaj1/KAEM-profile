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
    "DDPM": "#9b59b6",
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


def _scale_spans_decades(values: list[float], threshold: float = 5.0) -> bool:
    """True if max(values) / min(positive values) exceeds threshold."""
    pos = [v for v in values if v > 0]
    if len(pos) < 2:
        return False
    return max(pos) / min(pos) > threshold


def plot_grouped_bars(
    entries: list[tuple[pd.DataFrame, str]],
    title: str,
    output_name: str,
    references: list[tuple[pd.DataFrame, str]] | None = None,
):
    """Plot grouped bars across models, with constant-cost references as
    horizontal dashed lines. entries / references: list of (dataframe, label).
    Switches a panel to log-y when the values span more than one order of
    magnitude (e.g. when a DDPM reference dwarfs the bars).
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    references = references or []

    latent_dims = sorted(entries[0][0]["Latent Dim"].unique())
    n_models = len(entries)
    width = 0.8 / n_models
    x = np.arange(len(latent_dims))

    for idx, (metric, metric_title) in enumerate(zip(METRICS, METRIC_TITLES)):
        ax = axs[idx]
        all_values: list[float] = []

        for m_idx, (df, label) in enumerate(entries):
            offset = (m_idx - (n_models - 1) / 2) * width
            values = [
                df[df["Latent Dim"] == ld][metric].values[0] for ld in latent_dims
            ]
            all_values.extend(float(v) for v in values)
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

        for df, label in references:
            value = float(df[metric].iloc[0])
            all_values.append(value)
            ax.axhline(
                value,
                color=PALETTE[label],
                linestyle="--",
                linewidth=1.8,
                alpha=0.9,
                label=label,
            )

        if _scale_spans_decades(all_values):
            ax.set_yscale("log")

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
        ncol=len(handles),
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
    references: list[tuple[pd.DataFrame, str]] | None = None,
):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    references = references or []
    latent_dims = sorted(entries[0][0]["Latent Dim"].unique())
    n_models = len(entries)
    width = 0.8 / n_models
    x = np.arange(len(latent_dims))
    all_values: list[float] = []

    for m_idx, (df, label) in enumerate(entries):
        offset = (m_idx - (n_models - 1) / 2) * width
        times = [df[df["Latent Dim"] == ld]["Time (s)"].values[0] for ld in latent_dims]
        stds = [
            df[df["Latent Dim"] == ld]["Time Std (s)"].values[0] for ld in latent_dims
        ]
        all_values.extend(float(t) for t in times)
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

    for df, label in references:
        value = float(df["Time (s)"].iloc[0])
        all_values.append(value)
        ax.axhline(
            value,
            color=PALETTE[label],
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label=label,
        )

    if _scale_spans_decades(all_values):
        ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels(latent_dims, fontsize=15)
    ax.set_xlabel(r"Latent Dim, $Q$", fontsize=18)
    ax.set_ylabel("Time (s)", fontsize=18)
    ax.set_title(title, fontsize=19, pad=10)
    ax.legend(
        fontsize=13,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )
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
    kaem_train_ula = load_csv_safe(RESULTS_DIR / "latent_dim_ula.csv")
    kaem_train_importance = load_csv_safe(RESULTS_DIR / "latent_dim_importance.csv")
    kaem_sample = load_csv_safe(RESULTS_DIR / "ITS_generation.csv")
    temperatures = load_csv_safe(RESULTS_DIR / "temperatures.csv")

    vae_train = load_csv_safe(RESULTS_DIR / "vae_latent_dim.csv")
    vae_sample = load_csv_safe(RESULTS_DIR / "vae_sampling.csv")

    pang_train = load_csv_safe(RESULTS_DIR / "pang_latent_dim.csv")
    pang_sample = load_csv_safe(RESULTS_DIR / "pang_sampling.csv")

    ddpm_train = load_csv_safe(RESULTS_DIR / "ddpm_latent_dim.csv")
    ddpm_sample = load_csv_safe(RESULTS_DIR / "ddpm_sampling.csv")

    train_refs = []
    if ddpm_train is not None:
        train_refs.append((normalize(ddpm_train, "latent_dim", "DDPM", False), "DDPM"))

    def baseline_train_entries():
        entries = []
        if vae_train is not None:
            entries.append((normalize(vae_train, "latent_dim", "VAE", False), "VAE"))
        if pang_train is not None:
            entries.append((normalize(pang_train, "latent_dim", "Pang", False), "Pang"))
        return entries

    # Training comparison — KAEM with ULA posterior sampling
    if kaem_train_ula is not None:
        ula_entries = [
            (normalize(kaem_train_ula, "n_z", "KAEM", True), "KAEM"),
            *baseline_train_entries(),
        ]
        if len(ula_entries) >= 2:
            plot_grouped_bars(
                ula_entries,
                "Training Cost (KAEM with ULA)",
                "01_training_comparison.png",
                references=train_refs,
            )
            print("Saved: 01_training_comparison.png")

            plot_time_only(
                ula_entries,
                "Training Time (KAEM with ULA)",
                "01b_training_time_only.png",
                references=train_refs,
            )
            print("Saved: 01b_training_time_only.png")

    # Training comparison, KAEM with importance sampling
    if kaem_train_importance is not None:
        is_entries = [
            (normalize(kaem_train_importance, "n_z", "KAEM", True), "KAEM"),
            *baseline_train_entries(),
        ]
        if len(is_entries) >= 2:
            plot_grouped_bars(
                is_entries,
                "Training Cost (KAEM with Importance Sampling)",
                "01_training_comparison_importance.png",
                references=train_refs,
            )
            print("Saved: 01_training_comparison_importance.png")

            plot_time_only(
                is_entries,
                "Training Time (KAEM with Importance Sampling)",
                "01b_training_time_only_importance.png",
                references=train_refs,
            )
            print("Saved: 01b_training_time_only_importance.png")

    # Sampling comparison
    sample_entries = []
    sample_refs = []
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
    if ddpm_sample is not None:
        sample_refs.append(
            (normalize(ddpm_sample, "latent_dim", "DDPM", False), "DDPM")
        )

    if len(sample_entries) >= 2:
        plot_grouped_bars(
            sample_entries,
            "Sampling Cost",
            "02_sampling_comparison.png",
            references=sample_refs,
        )
        print("Saved: 02_sampling_comparison.png")

        plot_time_only(
            sample_entries,
            "Sampling Time",
            "02b_sampling_time_only.png",
            references=sample_refs,
        )
        print("Saved: 02b_sampling_time_only.png")

    # Power posteriors
    if temperatures is not None:
        plot_temperatures(temperatures, "03_training_vs_num_temperatures.png")
        print("Saved: 03_training_vs_num_temperatures.png")


if __name__ == "__main__":
    main()
