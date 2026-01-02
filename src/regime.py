import pandas as pd
import numpy as np

from . import config

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


def _fit_hmm(X: np.ndarray, n_components: int) -> GaussianHMM:
    covar_types = [config.HMM_COVARIANCE_TYPE or "full"]
    if covar_types[0] == "full":
        covar_types.append("diag")
    last_err: Exception | None = None
    for covar_type in covar_types:
        try:
            model = GaussianHMM(
                n_components=n_components,
                covariance_type=covar_type,
                n_iter=100,
                min_covar=config.HMM_MIN_COVAR,
                random_state=config.SEED,
            )
            model.fit(X)
            return model
        except Exception as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    raise RuntimeError("HMM fit failed unexpectedly.")


def _apply_heuristic_trend_vol(market_proxy: pd.DataFrame) -> None:
    market_proxy["trend_up"] = market_proxy["sma_50"] > market_proxy["sma_200"]
    vol_threshold = (
        market_proxy["vol"]
        .shift(1)
        .expanding(min_periods=config.ROLLING_WINDOW_FOR_VOL)
        .quantile(0.75)
    )
    market_proxy["vol_high"] = market_proxy["vol"] > vol_threshold


def _apply_hmm_trend_vol(market_proxy: pd.DataFrame) -> None:
    if not HMM_AVAILABLE:
        raise ImportError("hmmlearn is required for REGIME_MODE=hmm. Please install hmmlearn.")
    data = market_proxy[["market_return", "vol"]].dropna()
    if len(data) < max(50, config.HMM_N_COMPONENTS * 10):
        _apply_heuristic_trend_vol(market_proxy)
        return
    X = np.column_stack([data["market_return"].to_numpy(), data["vol"].to_numpy()])
    try:
        model = _fit_hmm(X, config.HMM_N_COMPONENTS)
    except Exception:
        _apply_heuristic_trend_vol(market_proxy)
        return
    states = model.predict(X)
    state_series = pd.Series(states, index=data.index, dtype="Int64")
    market_proxy["hmm_state"] = pd.Series(index=market_proxy.index, dtype="Int64")
    market_proxy.loc[state_series.index, "hmm_state"] = state_series

    stats = []
    for i in range(config.HMM_N_COMPONENTS):
        mask = states == i
        if not np.any(mask):
            avg_ret = 0.0
            avg_vol = 0.0
        else:
            avg_ret = float(np.mean(X[mask, 0]))
            avg_vol = float(np.mean(X[mask, 1]))
        stats.append({"state": i, "avg_ret": avg_ret, "avg_vol": avg_vol})
    stats_df = pd.DataFrame(stats)
    ret_threshold = stats_df["avg_ret"].median()
    vol_threshold = stats_df["avg_vol"].median()
    bull_states = set(stats_df[stats_df["avg_ret"] > ret_threshold]["state"].tolist())
    high_vol_states = set(stats_df[stats_df["avg_vol"] > vol_threshold]["state"].tolist())

    trend_up_series = pd.Series(states, index=data.index).isin(bull_states)
    vol_high_series = pd.Series(states, index=data.index).isin(high_vol_states)

    _apply_heuristic_trend_vol(market_proxy)
    market_proxy.loc[trend_up_series.index, "trend_up"] = trend_up_series
    market_proxy.loc[vol_high_series.index, "vol_high"] = vol_high_series


def _apply_hmm_rolling_trend_vol(market_proxy: pd.DataFrame) -> None:
    if not HMM_AVAILABLE:
        raise ImportError("hmmlearn is required for REGIME_MODE=hmm_rolling. Please install hmmlearn.")
    data = market_proxy[["market_return", "vol"]].dropna()
    if data.empty:
        _apply_heuristic_trend_vol(market_proxy)
        return

    warmup = max(50, config.HMM_WARMUP_PERIOD)
    step = max(1, config.HMM_STEP_SIZE)
    min_train = max(50, config.HMM_N_COMPONENTS * 10)

    trend_up = pd.Series(index=data.index, dtype=bool)
    vol_high = pd.Series(index=data.index, dtype=bool)
    hmm_state = pd.Series(index=data.index, dtype="Int64")

    train_end = warmup
    while train_end < len(data):
        train_slice = data.iloc[:train_end]
        predict_slice = data.iloc[train_end : train_end + step]
        if len(train_slice) < min_train or predict_slice.empty:
            train_end += step
            continue

        X_train = np.column_stack([train_slice["market_return"].to_numpy(), train_slice["vol"].to_numpy()])
        X_pred = np.column_stack([predict_slice["market_return"].to_numpy(), predict_slice["vol"].to_numpy()])
        try:
            model = _fit_hmm(X_train, config.HMM_N_COMPONENTS)
            train_states = model.predict(X_train)
            pred_states = model.predict(X_pred)

            stats = []
            for i in range(config.HMM_N_COMPONENTS):
                mask = train_states == i
                if not np.any(mask):
                    avg_ret = 0.0
                    avg_vol = 0.0
                else:
                    avg_ret = float(np.mean(X_train[mask, 0]))
                    avg_vol = float(np.mean(X_train[mask, 1]))
                stats.append({"state": i, "avg_ret": avg_ret, "avg_vol": avg_vol})
            stats_df = pd.DataFrame(stats)
            ret_threshold = stats_df["avg_ret"].median()
            vol_threshold = stats_df["avg_vol"].median()
            bull_states = set(stats_df[stats_df["avg_ret"] > ret_threshold]["state"].tolist())
            high_vol_states = set(stats_df[stats_df["avg_vol"] > vol_threshold]["state"].tolist())

            trend_up.loc[predict_slice.index] = pd.Series(pred_states, index=predict_slice.index).isin(bull_states)
            vol_high.loc[predict_slice.index] = pd.Series(pred_states, index=predict_slice.index).isin(high_vol_states)
            hmm_state.loc[predict_slice.index] = pd.Series(pred_states, index=predict_slice.index)
        except Exception:
            pass

        train_end += step

    _apply_heuristic_trend_vol(market_proxy)
    market_proxy["hmm_state"] = pd.Series(index=market_proxy.index, dtype="Int64")
    market_proxy.loc[hmm_state.index, "hmm_state"] = hmm_state
    if not trend_up.empty:
        market_proxy.loc[trend_up.index, "trend_up"] = trend_up
    if not vol_high.empty:
        market_proxy.loc[vol_high.index, "vol_high"] = vol_high


def compute_market_regime_table(df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """
    Build a per-date regime table using:
      - Trend: market SMA50 vs SMA200 (market = average close across universe)
      - Volatility: rolling std of market returns vs 75th percentile (heuristic)
      - Breadth: % of stocks with Close > SMA_50
      - Dispersion: cross-sectional std of ROC_10_z (as a proxy for momentum dispersion)
    Returns a DataFrame indexed by date with booleans and helper labels.
    """
    mode = (mode or config.REGIME_MODE).lower()
    market_proxy = df.groupby(level="date")["Close"].mean().to_frame("market_close")
    market_proxy["market_return"] = market_proxy["market_close"].pct_change()
    market_proxy["sma_50"] = market_proxy["market_close"].rolling(window=50, min_periods=50).mean()
    market_proxy["sma_200"] = market_proxy["market_close"].rolling(window=200, min_periods=200).mean()
    market_proxy["vol"] = market_proxy["market_return"].rolling(
        window=config.ROLLING_WINDOW_FOR_VOL,
        min_periods=config.ROLLING_WINDOW_FOR_VOL,
    ).std()
    market_proxy["hmm_state"] = pd.Series(index=market_proxy.index, dtype="Int64")

    if mode == "hmm":
        _apply_hmm_trend_vol(market_proxy)
    elif mode == "hmm_rolling":
        _apply_hmm_rolling_trend_vol(market_proxy)
    else:
        _apply_heuristic_trend_vol(market_proxy)

    # Breadth: share of stocks above their SMA_50 (neutral 0.5 if SMA_50 missing)
    if "SMA_50" in df.columns:
        breadth = (df["Close"] > df["SMA_50"]).groupby(level="date").mean()
        breadth = breadth.reindex(market_proxy.index)
    else:
        breadth = pd.Series(0.5, index=market_proxy.index)
    market_proxy["breadth"] = breadth
    market_proxy["breadth_low"] = market_proxy["breadth"] < 0.45
    market_proxy["breadth_high"] = market_proxy["breadth"] > 0.55

    # Dispersion: cross-sectional std of ROC_10_z (neutral 0 if missing)
    if "ROC_10_z" in df.columns:
        dispersion = df.groupby(level="date")["ROC_10_z"].std()
        dispersion = dispersion.reindex(market_proxy.index)
    else:
        dispersion = pd.Series(0.0, index=market_proxy.index)
    dispersion_high_q = dispersion.shift(1).expanding(min_periods=30).quantile(0.75)
    dispersion_low_q = dispersion.shift(1).expanding(min_periods=30).quantile(0.25)
    market_proxy["dispersion"] = dispersion
    market_proxy["dispersion_high"] = market_proxy["dispersion"] > dispersion_high_q
    market_proxy["dispersion_low"] = market_proxy["dispersion"] < dispersion_low_q

    # Combined regime label (trend x vol)
    def _label(row):
        if row["trend_up"] and not row["vol_high"]:
            return "bull_low_vol"
        if row["trend_up"] and row["vol_high"]:
            return "bull_high_vol"
        if (not row["trend_up"]) and row["vol_high"]:
            return "bear_high_vol"
        return "bear_low_vol"

    market_proxy["regime_label"] = market_proxy.apply(_label, axis=1)
    if (
        config.HMM_STATE_LABELS
        and mode in ("hmm", "hmm_rolling")
        and "hmm_state" in market_proxy.columns
    ):
        def _label_hmm(state):
            if pd.isna(state):
                return "unknown"
            return f"hmm_state_{int(state)}"

        market_proxy["regime_label"] = market_proxy["hmm_state"].apply(_label_hmm)
    market_proxy["combined_regime"] = 2 * market_proxy["trend_up"].astype(float) + market_proxy["vol_high"].astype(float)
    return market_proxy[
        [
            "trend_up",
            "vol_high",
            "hmm_state",
            "breadth",
            "breadth_low",
            "breadth_high",
            "dispersion",
            "dispersion_high",
            "dispersion_low",
            "regime_label",
            "combined_regime",
        ]
    ]


def get_regime_state(regime_table: pd.DataFrame | None, current_date) -> dict:
    if regime_table is None or current_date not in regime_table.index:
        return {
            "trend_up": False,
            "vol_high": False,
            "dispersion_high": False,
            "dispersion_low": False,
            "breadth_low": False,
            "breadth_high": False,
            "regime_label": "unknown",
        }
    row = regime_table.loc[current_date]
    return {
        "trend_up": bool(row.get("trend_up", False)),
        "vol_high": bool(row.get("vol_high", False)),
        "dispersion_high": bool(row.get("dispersion_high", False)),
        "dispersion_low": bool(row.get("dispersion_low", False)),
        "breadth_low": bool(row.get("breadth_low", False)),
        "breadth_high": bool(row.get("breadth_high", False)),
        "regime_label": str(row.get("regime_label", "unknown")),
    }


def regime_top_k(state: dict, default_top_k: int) -> int:
    if state.get("vol_high") and state.get("dispersion_low") and state.get("breadth_low"):
        return 5
    if state.get("vol_high") and state.get("dispersion_high"):
        return 7
    if (not state.get("vol_high")) and state.get("dispersion_high") and state.get("breadth_high"):
        return 12
    if (not state.get("vol_high")) and state.get("dispersion_high"):
        return 10
    if state.get("vol_high"):
        return 6
    return default_top_k


def regime_gross_target(state: dict) -> float:
    trend_up = bool(state.get("trend_up", False))
    vol_high = bool(state.get("vol_high", False))
    dispersion_high = bool(state.get("dispersion_high", False))
    dispersion_low = bool(state.get("dispersion_low", False))
    breadth_low = bool(state.get("breadth_low", False))
    breadth_high = bool(state.get("breadth_high", False))

    gross = 0.85
    if (not trend_up) and vol_high:
        gross = 0.6
    elif (not trend_up) and (not vol_high):
        gross = 0.75
    elif vol_high and dispersion_low and breadth_low:
        gross = 0.6
    elif vol_high and dispersion_high:
        gross = 0.8
    elif (not vol_high) and dispersion_high and breadth_high:
        gross = 1.05
    elif (not vol_high) and dispersion_high:
        gross = 0.95
    elif (not vol_high) and dispersion_low:
        gross = 0.85
    return max(0.0, min(1.1, gross))
