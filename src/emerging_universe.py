from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmergingUniverseParams:
    enabled: bool = False
    rebalance_frequency: str = "MS"
    top_n: int = 150
    min_history_days: int = 252
    min_price: float = 20.0
    min_adv_rank: float = 0.05
    max_adv_rank: float = 0.85
    min_ret_6m: float = 0.10
    min_ret_12m: float = -0.10
    min_adv_growth_6m: float = 1.05


@dataclass
class EmergingUniverseSchedule:
    params: EmergingUniverseParams
    members_by_date: dict[pd.Timestamp, set[str]]
    rebalance_dates: list[pd.Timestamp]
    diagnostics: pd.DataFrame

    def allowed_tickers_for_date(self, date: pd.Timestamp) -> set[str] | None:
        key = pd.Timestamp(date).tz_localize(None)
        return self.members_by_date.get(key)


def _to_naive_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts


def _normalize_bounds(min_rank: float, max_rank: float) -> tuple[float, float]:
    lo = float(np.clip(min_rank, 0.0, 1.0))
    hi = float(np.clip(max_rank, 0.0, 1.0))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _cs_zscore(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    vals = pd.to_numeric(series, errors="coerce")
    mean = vals.mean()
    std = vals.std(ddof=0)
    if pd.isna(std) or float(std) <= 1e-9:
        return pd.Series(0.0, index=series.index)
    return (vals - float(mean)) / float(std)


def _select_rebalance_dates(
    trading_dates: pd.Index,
    frequency: str,
) -> list[pd.Timestamp]:
    if len(trading_dates) == 0:
        return []
    freq = str(frequency or "MS").strip().upper()
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).tz_localize(None)
    if freq in {"MS", "MONTHLY", "MONTH_START"}:
        groups = pd.Series(dates, index=dates).groupby(dates.to_period("M"))
        return [_to_naive_ts(group.iloc[0]) for _, group in groups]
    if freq in {"W", "WEEKLY"}:
        groups = pd.Series(dates, index=dates).groupby(dates.to_period("W"))
        return [_to_naive_ts(group.iloc[0]) for _, group in groups]
    if freq.endswith("D"):
        try:
            step = max(1, int(freq[:-1]))
        except ValueError:
            step = 21
        return [_to_naive_ts(dates[i]) for i in range(0, len(dates), step)]
    return [_to_naive_ts(dates[0])]


def build_emerging_universe_schedule(
    df: pd.DataFrame,
    params: EmergingUniverseParams,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> EmergingUniverseSchedule:
    if not params.enabled:
        return EmergingUniverseSchedule(
            params=params,
            members_by_date={},
            rebalance_dates=[],
            diagnostics=pd.DataFrame(),
        )
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex [date, ticker] for emerging universe schedule.")
    if "Close" not in df.columns or "adv_21" not in df.columns:
        raise ValueError("Data must include Close and adv_21 columns for emerging universe schedule.")

    work = df[["Close", "adv_21"]].copy().sort_index()
    work["ticker"] = work.index.get_level_values("ticker").astype(str).str.upper()
    grouped = work.groupby(level="ticker", sort=False)

    work["ret_6m"] = grouped["Close"].pct_change(126)
    work["ret_12m"] = grouped["Close"].pct_change(252)
    work["adv_growth_6m"] = work["adv_21"] / (grouped["adv_21"].shift(126) + 1e-9)
    work["history_days"] = grouped.cumcount() + 1
    work["adv_rank"] = work.groupby(level="date")["adv_21"].rank(pct=True)

    work["ret_6m_z"] = work.groupby(level="date")["ret_6m"].transform(_cs_zscore)
    work["ret_12m_z"] = work.groupby(level="date")["ret_12m"].transform(_cs_zscore)
    work["adv_growth_6m_z"] = work.groupby(level="date")["adv_growth_6m"].transform(_cs_zscore)
    work["emerging_score"] = (
        0.50 * work["ret_6m_z"]
        + 0.25 * work["ret_12m_z"]
        + 0.25 * work["adv_growth_6m_z"]
    )

    all_dates = pd.Index(work.index.get_level_values("date").unique()).sort_values()
    if start_date is not None:
        all_dates = all_dates[all_dates >= _to_naive_ts(start_date)]
    if end_date is not None:
        all_dates = all_dates[all_dates < _to_naive_ts(end_date)]

    rebalance_dates = _select_rebalance_dates(all_dates, params.rebalance_frequency)
    lo_rank, hi_rank = _normalize_bounds(params.min_adv_rank, params.max_adv_rank)

    members_by_date: dict[pd.Timestamp, set[str]] = {}
    diagnostics_rows: list[dict] = []
    previous_members: set[str] = set()
    all_dates_list = [pd.Timestamp(d).tz_localize(None) for d in all_dates]
    date_to_pos = {d: i for i, d in enumerate(all_dates_list)}

    for i, rebalance_date in enumerate(rebalance_dates):
        if rebalance_date not in date_to_pos:
            continue
        try:
            day = work.xs(rebalance_date, level="date")
        except KeyError:
            continue
        if day.empty:
            continue
        day = day.copy()
        day.index = day.index.astype(str).str.upper()
        finite_score = pd.to_numeric(day["emerging_score"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        day["emerging_score"] = finite_score.fillna(-999.0)

        eligible = (
            (pd.to_numeric(day["Close"], errors="coerce") >= float(params.min_price))
            & (pd.to_numeric(day["history_days"], errors="coerce") >= int(params.min_history_days))
            & (pd.to_numeric(day["adv_rank"], errors="coerce") >= lo_rank)
            & (pd.to_numeric(day["adv_rank"], errors="coerce") <= hi_rank)
            & (pd.to_numeric(day["ret_6m"], errors="coerce") >= float(params.min_ret_6m))
            & (pd.to_numeric(day["ret_12m"], errors="coerce") >= float(params.min_ret_12m))
            & (pd.to_numeric(day["adv_growth_6m"], errors="coerce") >= float(params.min_adv_growth_6m))
        )
        candidate = day[eligible].sort_values("emerging_score", ascending=False)
        selected = set(candidate.head(int(max(1, params.top_n))).index.tolist())
        if not selected:
            selected = set(previous_members)
        if not selected:
            fallback = day[pd.to_numeric(day["Close"], errors="coerce") > 0]
            selected = set(fallback.sort_values("adv_21", ascending=False).head(50).index.tolist())

        diagnostics_rows.append(
            {
                "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                "eligible_count": int(np.sum(eligible)),
                "selected_count": int(len(selected)),
                "score_median_selected": float(candidate["emerging_score"].head(int(max(1, params.top_n))).median())
                if not candidate.empty
                else np.nan,
                "score_median_eligible": float(candidate["emerging_score"].median()) if not candidate.empty else np.nan,
            }
        )

        start_pos = date_to_pos[rebalance_date]
        end_pos = date_to_pos.get(rebalance_dates[i + 1], len(all_dates_list)) if i + 1 < len(rebalance_dates) else len(all_dates_list)
        for j in range(start_pos, end_pos):
            members_by_date[all_dates_list[j]] = set(selected)
        previous_members = set(selected)

    diagnostics = pd.DataFrame(diagnostics_rows)
    return EmergingUniverseSchedule(
        params=params,
        members_by_date=members_by_date,
        rebalance_dates=[_to_naive_ts(d) for d in rebalance_dates],
        diagnostics=diagnostics,
    )


def normalize_allowed_tickers_by_date(
    mapping: Mapping[pd.Timestamp, set[str]] | None,
) -> dict[pd.Timestamp, set[str]]:
    if not mapping:
        return {}
    out: dict[pd.Timestamp, set[str]] = {}
    for date_key, tickers in mapping.items():
        key = _to_naive_ts(date_key)
        out[key] = {str(t).strip().upper() for t in (tickers or set()) if str(t).strip()}
    return out
