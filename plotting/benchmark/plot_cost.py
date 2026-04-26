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
    "Neural Latent EBM": "#e67e22",
    "DDPM": "#7f1d1d",
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


def _format_time(v: float) -> str:
    if v < 1e-3:
        return rf"{v * 1e6:.0f}\,$\mu$s"
    if v < 1.0:
        return rf"{v * 1e3:.1f}\,ms"
    return rf"{v:.2f}\,s"


def _format_memory(v_gib: float) -> str:
    bytes_ = v_gib * (1024**3)
    if bytes_ < 1024:
        return rf"{bytes_:.0f}\,B"
    if bytes_ < 1024**2:
        return rf"{bytes_ / 1024:.1f}\,KiB"
    if bytes_ < 1024**3:
        return rf"{bytes_ / 1024**2:.1f}\,MiB"
    return rf"{v_gib:.2f}\,GiB"


def _format_count(v: float) -> str:
    return f"{int(round(v))}"


def _formatter(metric: str):
    if metric == "Time (s)":
        return _format_time
    if metric == "Memory Estimate (GiB)":
        return _format_memory
    return _format_count


_BREAK_RATIO = 5.0
_HEIGHT_RATIO = 3.5  # bottom : top for broken-axis panels


def _draw_break_marks(ax_top, ax_bot, height_ratio: float = _HEIGHT_RATIO):
    """Draw diagonal break marks at the top of ax_bot and bottom of ax_top."""
    d = 0.015
    d_bot = d / height_ratio
    kw_top = dict(transform=ax_top.transAxes, color="black", clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kw_top)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kw_top)
    kw_bot = dict(transform=ax_bot.transAxes, color="black", clip_on=False, linewidth=1)
    ax_bot.plot((-d, +d), (1 - d_bot, 1 + d_bot), **kw_bot)
    ax_bot.plot((1 - d, 1 + d), (1 - d_bot, 1 + d_bot), **kw_bot)


def _draw_bars_on_axis(
    ax,
    entries,
    metric: str,
    latent_dims,
    width: float,
    x,
    fmt,
    legend_handles: dict,
):
    n_models = len(entries)
    for m_idx, (df, label) in enumerate(entries):
        offset = (m_idx - (n_models - 1) / 2) * width
        values = [
            float(df[df["Latent Dim"] == ld][metric].values[0]) for ld in latent_dims
        ]
        kwargs = {
            "label": label,
            "color": PALETTE[label],
            "edgecolor": "white",
            "linewidth": 0.5,
        }
        if metric == "Time (s)" and "Time Std (s)" in df.columns:
            stds = [
                float(df[df["Latent Dim"] == ld]["Time Std (s)"].values[0])
                for ld in latent_dims
            ]
            bars = ax.bar(
                x + offset,
                values,
                width,
                yerr=stds,
                capsize=3,
                error_kw={"elinewidth": 1.2, "capthick": 1.2, "alpha": 0.8},
                **kwargs,
            )
        else:
            bars = ax.bar(x + offset, values, width, **kwargs)

        ax.bar_label(
            bars,
            labels=[fmt(float(v)) for v in values],
            padding=3,
            fontsize=11,
            rotation=90,
        )
        legend_handles.setdefault(label, bars)


def _draw_reference_on_axis(
    ax, references, metric: str, fmt, legend_handles: dict, label_fontsize: int
):
    for df, label in references:
        value = float(df[metric].iloc[0])
        line = ax.axhline(
            value,
            color=PALETTE[label],
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label=label,
        )
        ax.text(
            -0.5,
            value,
            rf"{label}: {fmt(value)} ",
            color=PALETTE[label],
            fontsize=label_fontsize,
            verticalalignment="bottom",
            horizontalalignment="left",
        )
        legend_handles.setdefault(label, line)


def plot_grouped_bars(
    entries: list[tuple[pd.DataFrame, str]],
    title: str,
    output_name: str,
    references: list[tuple[pd.DataFrame, str]] | None = None,
):
    """Plot grouped bars with constant-cost references"""
    references = references or []
    fig = plt.figure(figsize=(24, 6.0))
    outer_gs = fig.add_gridspec(1, 3, wspace=0.28)

    latent_dims = sorted(entries[0][0]["Latent Dim"].unique())
    n_models = len(entries)
    width = 0.8 / n_models
    x = np.arange(len(latent_dims))

    legend_handles: dict[str, object] = {}

    for idx, (metric, metric_title) in enumerate(zip(METRICS, METRIC_TITLES)):
        fmt = _formatter(metric)

        bar_values = [
            float(df[df["Latent Dim"] == ld][metric].values[0])
            for df, _ in entries
            for ld in latent_dims
        ]
        ref_values = [float(df[metric].iloc[0]) for df, _ in references]
        bar_max = max(bar_values) if bar_values else 0.0
        ref_max = max(ref_values, default=0.0)
        needs_break = bool(ref_values) and bar_max > 0 and ref_max > _BREAK_RATIO * bar_max

        if needs_break:
            sub = outer_gs[0, idx].subgridspec(
                2, 1, height_ratios=[1, _HEIGHT_RATIO], hspace=0.06
            )
            ax_top = fig.add_subplot(sub[0])
            ax_bot = fig.add_subplot(sub[1])
        else:
            ax_top = None
            ax_bot = fig.add_subplot(outer_gs[0, idx])

        _draw_bars_on_axis(
            ax_bot, entries, metric, latent_dims, width, x, fmt, legend_handles
        )
        target_ax = ax_top if needs_break else ax_bot
        _draw_reference_on_axis(target_ax, references, metric, fmt, legend_handles, 14)

        if needs_break:
            ax_bot.set_ylim(0, bar_max * 1.45)
            ref_min = min(ref_values)
            span = max(ref_max - ref_min, ref_max * 0.05)
            ax_top.set_ylim(ref_min - span * 0.6, ref_max + span * 0.6)
            ax_top.set_xlim(ax_bot.get_xlim())

            ax_bot.spines["top"].set_visible(False)
            ax_top.spines["bottom"].set_visible(False)
            ax_top.tick_params(
                axis="x", which="both", bottom=False, labelbottom=False
            )
            ax_top.set_xticks([])
            _draw_break_marks(ax_top, ax_bot)

            ax_top.set_title(metric_title, fontsize=22, pad=10)
            ax_top.tick_params(axis="y", labelsize=17)
            ax_top.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
            ax_top.set_axisbelow(True)
            ax_top.spines["right"].set_visible(False)
            ax_top.spines["top"].set_visible(False)
        else:
            ax_bot.set_title(metric_title, fontsize=22, pad=10)
            ax_bot.set_ylim(top=ax_bot.get_ylim()[1] * 1.45)
            ax_bot.spines["top"].set_visible(False)

        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(latent_dims, fontsize=18)
        ax_bot.set_xlabel(r"Latent Dim, $Q$", fontsize=21)
        ax_bot.set_ylabel(metric, fontsize=21)
        ax_bot.tick_params(axis="both", labelsize=17)
        ax_bot.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax_bot.set_axisbelow(True)
        ax_bot.spines["right"].set_visible(False)

    handles = list(legend_handles.values())
    labels = list(legend_handles.keys())
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
        fontsize=19,
        frameon=False,
    )
    fig.suptitle(title, fontsize=24, y=1.10)

    plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_only(
    entries: list[tuple[pd.DataFrame, str]],
    title: str,
    output_name: str,
    references: list[tuple[pd.DataFrame, str]] | None = None,
):
    """Single-panel time plot."""
    references = references or []
    latent_dims = sorted(entries[0][0]["Latent Dim"].unique())
    n_models = len(entries)
    width = 0.8 / n_models
    x = np.arange(len(latent_dims))
    fmt = _format_time

    times_all = [
        float(df[df["Latent Dim"] == ld]["Time (s)"].values[0])
        for df, _ in entries
        for ld in latent_dims
    ]
    ref_values = [float(df["Time (s)"].iloc[0]) for df, _ in references]
    bar_max = max(times_all) if times_all else 0.0
    ref_max = max(ref_values, default=0.0)
    needs_break = bool(ref_values) and bar_max > 0 and ref_max > _BREAK_RATIO * bar_max

    if needs_break:
        fig = plt.figure(figsize=(9, 6.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, _HEIGHT_RATIO], hspace=0.06)
        ax_top = fig.add_subplot(gs[0])
        ax_bot = fig.add_subplot(gs[1])
    else:
        fig, ax_bot = plt.subplots(figsize=(9, 5.5))
        ax_top = None

    legend_handles: dict[str, object] = {}

    for m_idx, (df, label) in enumerate(entries):
        offset = (m_idx - (n_models - 1) / 2) * width
        times = [
            float(df[df["Latent Dim"] == ld]["Time (s)"].values[0])
            for ld in latent_dims
        ]
        stds = [
            float(df[df["Latent Dim"] == ld]["Time Std (s)"].values[0])
            for ld in latent_dims
        ]
        bars = ax_bot.bar(
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
        ax_bot.bar_label(
            bars,
            labels=[fmt(float(t)) for t in times],
            padding=4,
            fontsize=10,
            rotation=45,
        )
        legend_handles.setdefault(label, bars)

    target_ax = ax_top if needs_break else ax_bot
    for df, label in references:
        value = float(df["Time (s)"].iloc[0])
        line = target_ax.axhline(
            value,
            color=PALETTE[label],
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label=label,
        )
        target_ax.text(
            -0.5,
            value,
            rf"{label}: {fmt(value)} ",
            color=PALETTE[label],
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="left",
        )
        legend_handles.setdefault(label, line)

    if needs_break:
        ax_bot.set_ylim(0, bar_max * 1.45)
        ref_min = min(ref_values)
        span = max(ref_max - ref_min, ref_max * 0.05)
        ax_top.set_ylim(ref_min - span * 0.6, ref_max + span * 0.6)
        ax_top.set_xlim(ax_bot.get_xlim())

        ax_bot.spines["top"].set_visible(False)
        ax_top.spines["bottom"].set_visible(False)
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_top.set_xticks([])
        _draw_break_marks(ax_top, ax_bot)

        ax_top.set_title(title, fontsize=19, pad=10)
        ax_top.tick_params(axis="y", labelsize=14)
        ax_top.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax_top.set_axisbelow(True)
        ax_top.spines["right"].set_visible(False)
        ax_top.spines["top"].set_visible(False)
    else:
        ax_bot.set_title(title, fontsize=19, pad=10)
        ax_bot.set_ylim(top=ax_bot.get_ylim()[1] * 1.18)
        ax_bot.spines["top"].set_visible(False)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(latent_dims, fontsize=15)
    ax_bot.set_xlabel(r"Latent Dim, $Q$", fontsize=18)
    ax_bot.set_ylabel("Time (s)", fontsize=18)
    ax_bot.tick_params(axis="both", labelsize=14)
    ax_bot.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax_bot.set_axisbelow(True)
    ax_bot.spines["right"].set_visible(False)

    handles = list(legend_handles.values())
    labels = list(legend_handles.keys())
    legend_target = ax_top if needs_break else ax_bot
    legend_target.legend(
        handles,
        labels,
        fontsize=13,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )

    plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_temperatures(df: pd.DataFrame, output_name: str):
    fig, axs = plt.subplots(1, 3, figsize=(24, 5.5))
    color = sns.color_palette("viridis", n_colors=4)[1]

    for idx, (metric, metric_title) in enumerate(zip(METRICS, METRIC_TITLES)):
        ax = axs[idx]
        fmt = _formatter(metric)
        col_map = {
            "Time (s)": "time_mean",
            "Memory Estimate (GiB)": "memory_estimate",
            "Allocations": "allocations",
        }
        values = df[col_map[metric]].values
        x = np.arange(len(df))

        if metric == "Time (s)":
            bars = ax.bar(
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
            bars = ax.bar(x, values, color=color, edgecolor="white", linewidth=0.5)

        # Drop redundant labels when every bar has the same value (e.g. memory
        # and allocations are flat across N_t in this benchmark).
        unique_values = {round(float(v), 12) for v in values}
        if len(unique_values) == 1:
            label_strings: list[str] = ["" for _ in values]
            label_strings[len(values) // 2] = fmt(float(values[0]))
        else:
            label_strings = [fmt(float(v)) for v in values]

        # Single-series chart with many bars — vertical labels avoid the
        # diagonal overlap that 45° rotation produces when bars are narrow.
        ax.bar_label(
            bars,
            labels=label_strings,
            padding=4,
            fontsize=14,
            rotation=90,
        )

        ax.set_ylim(top=ax.get_ylim()[1] * 1.45)

        ax.set_xticks(x)
        ax.set_xticklabels(df["N_t"].values, fontsize=18)
        ax.set_xlabel(r"$N_t$", fontsize=22)
        ax.set_ylabel(metric, fontsize=22)
        ax.set_title(metric_title, fontsize=24, pad=10)
        ax.tick_params(axis="both", labelsize=17)
        ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Power Posteriors", fontsize=26, y=1.02)
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
            entries.append(
                (
                    normalize(pang_train, "latent_dim", "Neural Latent EBM", False),
                    "Neural Latent EBM",
                )
            )
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
            (
                normalize(pang_sample, "latent_dim", "Neural Latent EBM", False),
                "Neural Latent EBM",
            )
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
