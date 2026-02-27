from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

TRADING_DAYS_PER_YEAR = 252.0
CAGR_NOTE = (
    "CAGR and cagr_adjusted are annualized on a 252-trading-day basis from summary rows; "
    "cagr_adjusted uses time-weighted returns with cash flows applied at period start; "
    "irr is money-weighted using net worth and cash adjustments."
)


def _xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float | None:
    if len(cashflows) < 2:
        return None
    cashflows = sorted(cashflows, key=lambda x: x[0])
    t0 = cashflows[0][0]

    def _year_frac(ts: pd.Timestamp) -> float:
        return (ts - t0).days / 365.25

    def npv(rate: float) -> float:
        total = 0.0
        for dt, amount in cashflows:
            total += amount / ((1.0 + rate) ** _year_frac(dt))
        return total

    def d_npv(rate: float) -> float:
        total = 0.0
        for dt, amount in cashflows:
            yf = _year_frac(dt)
            total += -yf * amount / ((1.0 + rate) ** (yf + 1.0))
        return total

    rate = 0.1
    for _ in range(50):
        f_val = npv(rate)
        if abs(f_val) < 1e-7:
            return rate
        deriv = d_npv(rate)
        if abs(deriv) < 1e-12:
            break
        rate -= f_val / deriv
        if rate <= -0.9999:
            rate = -0.9999

    low, high = -0.9999, 10.0
    f_low, f_high = npv(low), npv(high)
    if f_low * f_high > 0:
        return None
    mid = 0.0
    for _ in range(100):
        mid = (low + high) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-7:
            return mid
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return mid


def compute_cagr_summary(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for item in summaries:
        if not isinstance(item, dict):
            continue
        date_val = item.get("date")
        if not date_val:
            continue
        normalized.append(item)

    if len(normalized) < 2:
        return {
            "cagr": None,
            "cagr_adjusted": None,
            "detail": "Not enough history to compute CAGR.",
            "note": CAGR_NOTE,
        }

    normalized = sorted(normalized, key=lambda s: pd.to_datetime(s["date"]))
    start = normalized[0]
    end = normalized[-1]
    start_date = pd.to_datetime(start["date"])
    end_date = pd.to_datetime(end["date"])

    years = len(normalized) / TRADING_DAYS_PER_YEAR
    start_value = float(start.get("net_worth_usd", 0.0))
    end_value = float(end.get("net_worth_usd", 0.0))
    if years <= 0 or start_value <= 0:
        return {
            "cagr": None,
            "cagr_adjusted": None,
            "detail": "Invalid dates or starting value for CAGR calculation.",
            "note": CAGR_NOTE,
        }

    cagr = (end_value / start_value) ** (1.0 / years) - 1.0
    cash_adjustments = sum(float(s.get("cash_adjustment", 0.0)) for s in normalized)

    twr_growth = 1.0
    for i in range(1, len(normalized)):
        start_val = float(normalized[i - 1].get("net_worth_usd", 0.0))
        end_val = float(normalized[i].get("net_worth_usd", 0.0))
        flow = float(normalized[i].get("cash_adjustment", 0.0))
        denom = start_val + flow
        if denom <= 0:
            twr_growth = None
            break
        period_return = (end_val / denom) - 1.0
        twr_growth *= 1.0 + period_return

    cagr_adjusted = None
    if twr_growth is not None:
        cagr_adjusted = twr_growth ** (1.0 / years) - 1.0

    cashflows = [(start_date, -start_value)]
    for item in normalized[1:]:
        flow = float(item.get("cash_adjustment", 0.0))
        if flow != 0.0:
            cashflows.append((pd.to_datetime(item["date"]), -flow))
    cashflows.append((end_date, end_value))
    irr = _xirr(cashflows)

    broker_payload = None
    broker_summaries = [s for s in normalized if s.get("broker_net_worth") is not None]
    if len(broker_summaries) >= 2:
        broker_summaries = sorted(broker_summaries, key=lambda s: pd.to_datetime(s["date"]))
        b_start = broker_summaries[0]
        b_end = broker_summaries[-1]
        b_start_date = pd.to_datetime(b_start["date"])
        b_end_date = pd.to_datetime(b_end["date"])
        b_years = len(broker_summaries) / TRADING_DAYS_PER_YEAR
        b_start_val = float(b_start.get("broker_net_worth", 0.0))
        b_end_val = float(b_end.get("broker_net_worth", 0.0))
        if b_years > 0 and b_start_val > 0:
            b_cagr = (b_end_val / b_start_val) ** (1.0 / b_years) - 1.0
        else:
            b_cagr = None

        b_twr_growth = 1.0
        for i in range(1, len(broker_summaries)):
            start_val = float(broker_summaries[i - 1].get("broker_net_worth", 0.0))
            end_val = float(broker_summaries[i].get("broker_net_worth", 0.0))
            fx_rate = float(broker_summaries[i].get("broker_fx_rate_gbp_per_usd") or 0.0)
            flow_usd = float(broker_summaries[i].get("cash_adjustment", 0.0))
            flow = flow_usd * fx_rate if fx_rate else flow_usd
            denom = start_val + flow
            if denom <= 0:
                b_twr_growth = None
                break
            period_return = (end_val / denom) - 1.0
            b_twr_growth *= 1.0 + period_return

        b_cagr_adjusted = None
        if b_twr_growth is not None and b_years > 0:
            b_cagr_adjusted = b_twr_growth ** (1.0 / b_years) - 1.0

        b_cashflows = [(b_start_date, -b_start_val)]
        for item in broker_summaries[1:]:
            fx_rate = float(item.get("broker_fx_rate_gbp_per_usd") or 0.0)
            flow_usd = float(item.get("cash_adjustment", 0.0))
            flow = flow_usd * fx_rate if fx_rate else flow_usd
            if flow != 0.0:
                b_cashflows.append((pd.to_datetime(item["date"]), -flow))
        b_cashflows.append((b_end_date, b_end_val))
        b_irr = _xirr(b_cashflows)
        broker_payload = {
            "currency": b_start.get("broker_currency"),
            "cagr": b_cagr,
            "cagr_adjusted": b_cagr_adjusted,
            "irr": b_irr,
        }

    return {
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "years": years,
        "start_value_usd": start_value,
        "end_value_usd": end_value,
        "cagr": cagr,
        "cagr_adjusted": cagr_adjusted,
        "irr": irr,
        "cash_adjustments_total_usd": cash_adjustments,
        "broker": broker_payload,
        "note": CAGR_NOTE,
    }
