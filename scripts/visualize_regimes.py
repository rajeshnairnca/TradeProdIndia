import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src import config, regime


def _get_cmap(name: str, count: int):
    try:
        cmap = plt.colormaps.get_cmap(name)
        if hasattr(cmap, "resampled"):
            return cmap.resampled(count)
        return cmap
    except Exception:
        return plt.get_cmap(name, count)


def _resolve_data_path(region: str | None, data_file: str | None) -> str:
    if data_file:
        return data_file
    if region:
        return config.REGION_DATA_FILES.get(region.lower(), config.DATA_FILE)
    return config.DATA_FILE


def _plot_regime_4state(ax, market_proxy: pd.Series, reg_df: pd.DataFrame | None, title: str) -> None:
    ax.set_title(title)
    ax.set_ylabel("Average Price")
    ax.plot(market_proxy.index, market_proxy, "k-", linewidth=1)

    if reg_df is None:
        ax.text(0.5, 0.5, "Data Not Available", ha="center")
        return

    mask_bull_calm = (reg_df["trend_up"] & ~reg_df["vol_high"])
    mask_bull_vol = (reg_df["trend_up"] & reg_df["vol_high"])
    mask_bear_crash = (~reg_df["trend_up"] & reg_df["vol_high"])
    mask_bear_calm = (~reg_df["trend_up"] & ~reg_df["vol_high"])

    y_min, y_max = market_proxy.min(), market_proxy.max()
    dates = market_proxy.index
    ax.fill_between(dates, y_min, y_max, where=mask_bull_calm, color="green", alpha=0.2, label="Bull (Low Vol)")
    ax.fill_between(dates, y_min, y_max, where=mask_bull_vol, color="gold", alpha=0.3, label="Bull (High Vol)")
    ax.fill_between(dates, y_min, y_max, where=mask_bear_crash, color="red", alpha=0.3, label="Bear (High Vol)")
    ax.fill_between(dates, y_min, y_max, where=mask_bear_calm, color="orange", alpha=0.2, label="Bear (Low Vol)")
    ax.legend(loc="upper left")


def _plot_regime_states(ax, market_proxy: pd.Series, state_series: pd.Series, title: str) -> None:
    ax.set_title(title)
    ax.set_ylabel("Average Price")
    ax.plot(market_proxy.index, market_proxy, "k-", linewidth=1)
    if state_series.empty:
        ax.text(0.5, 0.5, "Data Not Available", ha="center")
        return

    y_min, y_max = market_proxy.min(), market_proxy.max()
    dates = market_proxy.index
    states = sorted([int(x) for x in pd.Series(state_series.dropna().unique()).astype(int).tolist()])
    if not states:
        ax.text(0.5, 0.5, "No HMM States", ha="center")
        return

    cmap = _get_cmap("tab10", max(3, len(states)))
    denom = max(1, len(states) - 1)
    for idx, state in enumerate(states):
        mask = state_series.eq(state).fillna(False)
        ax.fill_between(
            dates,
            y_min,
            y_max,
            where=mask,
            color=cmap(idx / denom),
            alpha=0.25,
            label=f"State {state}",
        )
    ax.legend(loc="upper left", ncol=2)


def plot_regime_comparison(
    region: str | None = None,
    data_file: str | None = None,
    output_file: str = "regime_comparison_4states.png",
    output_states_file: str = "regime_comparison_states.png",
    include_hmm: bool = True,
    include_hmm_rolling: bool = True,
    plot_hmm_states: bool = False,
) -> None:
    data_path = _resolve_data_path(region, data_file)
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)
    market_proxy = df.groupby(level="date")["Close"].mean()

    print("Computing Heuristic Regimes...")
    regime_heuristic = regime.compute_market_regime_table(df, mode="heuristic")

    regime_hmm = None
    if include_hmm:
        print(f"Computing HMM Regimes (Biased, N={config.HMM_N_COMPONENTS})...")
        try:
            regime_hmm = regime.compute_market_regime_table(df, mode="hmm")
        except ImportError as e:
            print(f"Skipping HMM: {e}")

    regime_hmm_rolling = None
    if include_hmm_rolling:
        print(
            "Computing HMM Rolling Regimes "
            f"(Unbiased, Step={config.HMM_STEP_SIZE}, N={config.HMM_N_COMPONENTS})..."
        )
        try:
            regime_hmm_rolling = regime.compute_market_regime_table(df, mode="hmm_rolling")
        except ImportError as e:
            print(f"Skipping HMM Rolling: {e}")

    common_idx = market_proxy.index.intersection(regime_heuristic.index)
    market_proxy = market_proxy.loc[common_idx]
    regime_heuristic = regime_heuristic.loc[common_idx]
    if regime_hmm is not None:
        regime_hmm = regime_hmm.reindex(common_idx)
    if regime_hmm_rolling is not None:
        regime_hmm_rolling = regime_hmm_rolling.reindex(common_idx)

    plots = [("heuristic", regime_heuristic, "Regime Detection: Heuristic (Hard Rules)")]
    if include_hmm:
        plots.append(("hmm", regime_hmm, f"Regime Detection: HMM (Full History, N={config.HMM_N_COMPONENTS})"))
    if include_hmm_rolling:
        plots.append(
            (
                "hmm_rolling",
                regime_hmm_rolling,
                f"Regime Detection: HMM Rolling (Unbiased, Step={config.HMM_STEP_SIZE}, N={config.HMM_N_COMPONENTS})",
            )
        )

    fig, axes = plt.subplots(len(plots), 1, figsize=(15, 5 * len(plots)), sharex=True)
    if len(plots) == 1:
        axes = [axes]
    for ax, (_, reg_df, title) in zip(axes, plots):
        _plot_regime_4state(ax, market_proxy, reg_df, title)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

    if plot_hmm_states and regime_hmm_rolling is not None:
        fig_states, ax_states = plt.subplots(1, 1, figsize=(15, 6), sharex=True)
        _plot_regime_states(
            ax_states,
            market_proxy,
            regime_hmm_rolling["hmm_state"].reindex(common_idx),
            f"HMM Rolling States (N={config.HMM_N_COMPONENTS})",
        )
        plt.tight_layout()
        plt.savefig(output_states_file)
        print(f"Saved HMM state plot to {output_states_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize heuristic and HMM regimes.")
    parser.add_argument("--region", type=str, default=None, help="Trading region (us/india).")
    parser.add_argument("--data-file", type=str, default=None, help="Override data file path.")
    parser.add_argument("--output", type=str, default="regime_comparison_4states.png", help="Output image file.")
    parser.add_argument(
        "--output-states",
        type=str,
        default="regime_comparison_states.png",
        help="Output image for HMM state view.",
    )
    parser.add_argument("--skip-hmm", action="store_true", help="Skip full-history HMM plot.")
    parser.add_argument("--skip-hmm-rolling", action="store_true", help="Skip HMM rolling plot.")
    parser.add_argument("--plot-hmm-states", action="store_true", help="Plot raw HMM state labels.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_regime_comparison(
        region=args.region,
        data_file=args.data_file,
        output_file=args.output,
        output_states_file=args.output_states,
        include_hmm=not args.skip_hmm,
        include_hmm_rolling=not args.skip_hmm_rolling,
        plot_hmm_states=args.plot_hmm_states,
    )
