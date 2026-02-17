from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import date as dt_date
from typing import Any, Iterable

import psycopg2
import psycopg2.extras

from .production import ProductionState

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or ""


def db_enabled() -> bool:
    return bool(DATABASE_URL.strip())


@contextmanager
def _connect():
    if not db_enabled():
        raise RuntimeError("DATABASE_URL/POSTGRES_URL not configured.")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    if not db_enabled():
        return
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS production_runs (
                run_date date PRIMARY KEY,
                created_at timestamptz DEFAULT now(),
                num_trades integer,
                net_worth_usd double precision,
                cash_usd double precision,
                portfolio_value_usd double precision,
                cash_weight double precision,
                regime text,
                strategies text,
                daily_costs_usd double precision,
                total_costs_usd double precision,
                sector text,
                regime_scope text,
                cash_adjustment double precision,
                broker_name text,
                broker_currency text,
                broker_cash double precision,
                broker_cash_before double precision,
                broker_cash_after double precision,
                broker_portfolio_value double precision,
                broker_net_worth double precision,
                broker_cash_weight double precision,
                broker_discrepancies jsonb,
                broker_buy_notional double precision,
                broker_sell_notional double precision,
                broker_external_flow double precision,
                broker_external_flow_usd double precision,
                broker_execution_cost double precision,
                broker_execution_cost_usd double precision,
                broker_total_execution_cost double precision,
                broker_total_execution_cost_usd double precision
            );
            """
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_name text;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_currency text;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_cash double precision;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_cash_before double precision;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_cash_after double precision;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_portfolio_value double precision;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_net_worth double precision;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_cash_weight double precision;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_discrepancies jsonb;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_buy_notional double precision;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_sell_notional double precision;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_external_flow double precision;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_external_flow_usd double precision;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_execution_cost double precision;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_execution_cost_usd double precision;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_total_execution_cost double precision;"
        )
        cur.execute(
            "ALTER TABLE production_runs ADD COLUMN IF NOT EXISTS broker_total_execution_cost_usd double precision;"
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS production_trades (
                id bigserial PRIMARY KEY,
                run_date date NOT NULL,
                ticker text,
                action text,
                shares double precision,
                price_usd double precision,
                value_usd double precision,
                net_worth_usd double precision,
                cash_usd double precision,
                portfolio_value_usd double precision,
                cash_weight double precision,
                regime text,
                strategies text,
                created_at timestamptz DEFAULT now()
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS production_prices (
                run_date date NOT NULL,
                ticker text NOT NULL,
                close_price double precision,
                created_at timestamptz DEFAULT now(),
                PRIMARY KEY (run_date, ticker)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS production_universe_map (
                ticker text PRIMARY KEY,
                exchange text,
                updated_at timestamptz DEFAULT now()
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS production_excluded_tickers (
                ticker text PRIMARY KEY,
                created_at timestamptz DEFAULT now()
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS production_run_calendar_overrides (
                run_date date PRIMARY KEY,
                action text NOT NULL CHECK (action IN ('skip', 'force_run')),
                reason text,
                source text,
                updated_at timestamptz DEFAULT now()
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS production_state (
                id integer PRIMARY KEY,
                last_date date,
                cash double precision,
                positions jsonb,
                prev_weights jsonb,
                total_costs_usd double precision,
                updated_at timestamptz DEFAULT now()
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS production_pending_adjustments (
                id bigserial PRIMARY KEY,
                payload jsonb NOT NULL,
                created_at timestamptz DEFAULT now()
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS production_broker_account (
                run_date date NOT NULL,
                broker text NOT NULL,
                currency text,
                cash double precision,
                investments double precision,
                net_worth double precision,
                payload jsonb,
                created_at timestamptz DEFAULT now(),
                PRIMARY KEY (run_date, broker)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS production_broker_positions (
                run_date date NOT NULL,
                broker text NOT NULL,
                ticker text NOT NULL,
                quantity double precision,
                average_price double precision,
                current_price double precision,
                instrument_currency text,
                account_currency text,
                wallet_current_value double precision,
                payload jsonb,
                created_at timestamptz DEFAULT now(),
                PRIMARY KEY (run_date, broker, ticker)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS production_broker_orders (
                id bigserial PRIMARY KEY,
                run_date date NOT NULL,
                broker text NOT NULL,
                ticker text,
                action text,
                quantity double precision,
                filled_quantity double precision,
                exec_price double precision,
                currency text,
                status text,
                order_id text,
                payload jsonb,
                created_at timestamptz DEFAULT now()
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS production_trades_run_date_idx ON production_trades (run_date);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS production_trades_ticker_idx ON production_trades (ticker);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS production_prices_run_date_idx ON production_prices (run_date);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS production_universe_map_exchange_idx ON production_universe_map (exchange);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS production_excluded_tickers_ticker_idx ON production_excluded_tickers (ticker);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS production_run_calendar_overrides_run_date_idx ON production_run_calendar_overrides (run_date);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS production_broker_orders_run_date_idx ON production_broker_orders (run_date);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS production_broker_orders_broker_idx ON production_broker_orders (broker);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS production_broker_positions_run_date_idx ON production_broker_positions (run_date);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS production_broker_account_run_date_idx ON production_broker_account (run_date);"
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS production_universe_monitor_state (
                id integer PRIMARY KEY,
                run_date date,
                generated_at_utc timestamptz,
                candidates_evaluated integer,
                tv_pass_count integer,
                tech_tv_pass_count integer,
                market_cap_pass_count integer,
                tech_after_market_cap_count integer,
                quality_pass_count integer,
                potential_additions_count integer,
                min_pass_days integer,
                payload jsonb,
                updated_at timestamptz DEFAULT now()
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS production_universe_monitor_candidates (
                ticker text PRIMARY KEY,
                run_date date,
                exchange text,
                symbol text,
                in_current_universe boolean,
                tv_data boolean,
                tv_pass boolean,
                tv_fail_reasons text,
                sector text,
                tech_sector boolean,
                quality_pass boolean,
                quality_reasons text,
                monitor_pass boolean,
                pass_streak integer,
                total_pass_days integer,
                close double precision,
                volume double precision,
                dollar_volume double precision,
                market_cap_basic double precision,
                recommend_all double precision,
                sma50 double precision,
                sma200 double precision,
                trend_up boolean,
                quality_rows double precision,
                quality_median_adv_dollars double precision,
                quality_p20_adv_dollars double precision,
                quality_p05_price double precision,
                quality_median_vol21 double precision,
                quality_p95_abs_log_return double precision,
                metadata_fail_reasons text,
                market_cap_pass boolean,
                yfinance_market_cap double precision,
                created_at timestamptz DEFAULT now()
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS production_universe_monitor_candidates_run_date_idx ON production_universe_monitor_candidates (run_date);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS production_universe_monitor_candidates_monitor_pass_idx ON production_universe_monitor_candidates (monitor_pass, pass_streak);"
        )


def upsert_run_summary(summary: dict[str, Any]) -> None:
    if not db_enabled():
        return
    init_db()
    run_date = summary.get("date")
    if not run_date:
        return
    strategies = summary.get("strategies", "")
    if isinstance(strategies, list):
        strategies = ",".join(strategies)
    daily_costs = summary.get("daily_costs_usd")
    if daily_costs is None:
        daily_costs = summary.get("total_costs_usd")
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO production_runs (
                run_date,
                num_trades,
                net_worth_usd,
                cash_usd,
                portfolio_value_usd,
                cash_weight,
                regime,
                strategies,
                daily_costs_usd,
                total_costs_usd,
                sector,
                regime_scope,
                cash_adjustment,
                broker_name,
                broker_currency,
                broker_cash,
                broker_cash_before,
                broker_cash_after,
                broker_portfolio_value,
                broker_net_worth,
                broker_cash_weight,
                broker_discrepancies,
                broker_buy_notional,
                broker_sell_notional,
                broker_external_flow,
                broker_external_flow_usd,
                broker_execution_cost,
                broker_execution_cost_usd,
                broker_total_execution_cost,
                broker_total_execution_cost_usd
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_date) DO UPDATE SET
                num_trades = EXCLUDED.num_trades,
                net_worth_usd = EXCLUDED.net_worth_usd,
                cash_usd = EXCLUDED.cash_usd,
                portfolio_value_usd = EXCLUDED.portfolio_value_usd,
                cash_weight = EXCLUDED.cash_weight,
                regime = EXCLUDED.regime,
                strategies = EXCLUDED.strategies,
                daily_costs_usd = EXCLUDED.daily_costs_usd,
                total_costs_usd = EXCLUDED.total_costs_usd,
                sector = EXCLUDED.sector,
                regime_scope = EXCLUDED.regime_scope,
                cash_adjustment = EXCLUDED.cash_adjustment,
                broker_name = EXCLUDED.broker_name,
                broker_currency = EXCLUDED.broker_currency,
                broker_cash = EXCLUDED.broker_cash,
                broker_cash_before = EXCLUDED.broker_cash_before,
                broker_cash_after = EXCLUDED.broker_cash_after,
                broker_portfolio_value = EXCLUDED.broker_portfolio_value,
                broker_net_worth = EXCLUDED.broker_net_worth,
                broker_cash_weight = EXCLUDED.broker_cash_weight,
                broker_discrepancies = EXCLUDED.broker_discrepancies,
                broker_buy_notional = EXCLUDED.broker_buy_notional,
                broker_sell_notional = EXCLUDED.broker_sell_notional,
                broker_external_flow = EXCLUDED.broker_external_flow,
                broker_external_flow_usd = EXCLUDED.broker_external_flow_usd,
                broker_execution_cost = EXCLUDED.broker_execution_cost,
                broker_execution_cost_usd = EXCLUDED.broker_execution_cost_usd,
                broker_total_execution_cost = EXCLUDED.broker_total_execution_cost,
                broker_total_execution_cost_usd = EXCLUDED.broker_total_execution_cost_usd;
            """,
            (
                run_date,
                summary.get("num_trades"),
                summary.get("net_worth_usd"),
                summary.get("cash_usd"),
                summary.get("portfolio_value_usd"),
                summary.get("cash_weight"),
                summary.get("regime"),
                strategies,
                daily_costs,
                summary.get("total_costs_usd"),
                summary.get("sector"),
                summary.get("regime_scope"),
                summary.get("cash_adjustment", 0.0),
                summary.get("broker_name"),
                summary.get("broker_currency"),
                summary.get("broker_cash"),
                summary.get("broker_cash_before"),
                summary.get("broker_cash_after"),
                summary.get("broker_portfolio_value"),
                summary.get("broker_net_worth"),
                summary.get("broker_cash_weight"),
                psycopg2.extras.Json(summary.get("broker_discrepancies"))
                if summary.get("broker_discrepancies") is not None
                else None,
                summary.get("broker_buy_notional"),
                summary.get("broker_sell_notional"),
                summary.get("broker_external_flow"),
                summary.get("broker_external_flow_usd"),
                summary.get("broker_execution_cost"),
                summary.get("broker_execution_cost_usd"),
                summary.get("broker_total_execution_cost"),
                summary.get("broker_total_execution_cost_usd"),
            ),
        )


def replace_trades(run_date: str, trades: Iterable[dict[str, Any]]) -> None:
    if not db_enabled():
        return
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM production_trades WHERE run_date = %s;", (run_date,))
        rows = []
        for trade in trades:
            rows.append(
                (
                    run_date,
                    trade.get("ticker"),
                    trade.get("action"),
                    trade.get("shares"),
                    trade.get("price_usd"),
                    trade.get("value_usd"),
                    trade.get("net_worth_usd"),
                    trade.get("cash_usd"),
                    trade.get("portfolio_value_usd"),
                    trade.get("cash_weight"),
                    trade.get("regime"),
                    trade.get("strategies"),
                )
            )
        if not rows:
            return
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO production_trades (
                run_date,
                ticker,
                action,
                shares,
                price_usd,
                value_usd,
                net_worth_usd,
                cash_usd,
                portfolio_value_usd,
                cash_weight,
                regime,
                strategies
            )
            VALUES %s;
            """,
            rows,
        )


def replace_prices(run_date: str, prices: Iterable[dict[str, Any]] | Iterable[tuple[str, float]]) -> None:
    if not db_enabled():
        return
    init_db()
    rows: list[tuple[str, str, float]] = []
    for item in prices:
        ticker = None
        price = None
        if isinstance(item, dict):
            ticker = item.get("ticker")
            price = item.get("close_price")
        else:
            try:
                ticker, price = item
            except (TypeError, ValueError):
                continue
        if ticker is None:
            continue
        ticker_str = str(ticker).strip().upper()
        if not ticker_str:
            continue
        try:
            price_val = float(price)
        except (TypeError, ValueError):
            continue
        rows.append((run_date, ticker_str, price_val))
    if not rows:
        return
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM production_prices WHERE run_date = %s;", (run_date,))
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO production_prices (
                run_date,
                ticker,
                close_price
            )
            VALUES %s;
            """,
            rows,
        )


def latest_prices(tickers: Iterable[str]) -> tuple[str | None, dict[str, float]]:
    if not db_enabled():
        return None, {}
    init_db()
    tickers_list = [str(t).strip().upper() for t in tickers if str(t).strip()]
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT run_date FROM production_prices ORDER BY run_date DESC LIMIT 1;")
        row = cur.fetchone()
        if not row or not row[0]:
            return None, {}
        run_date = row[0]
        if not tickers_list:
            return str(run_date), {}
        cur.execute(
            """
            SELECT ticker, close_price
            FROM production_prices
            WHERE run_date = %s AND ticker = ANY(%s);
            """,
            (run_date, tickers_list),
        )
        prices: dict[str, float] = {}
        for ticker, price in cur.fetchall():
            if ticker is None or price is None:
                continue
            prices[str(ticker).strip().upper()] = float(price)
        return str(run_date), prices


def latest_price_run_date() -> str | None:
    if not db_enabled():
        return None
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT run_date FROM production_prices ORDER BY run_date DESC LIMIT 1;")
        row = cur.fetchone()
        return str(row[0]) if row and row[0] else None


def price_tickers_for_date(run_date: str) -> set[str]:
    if not db_enabled():
        return set()
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ticker
            FROM production_prices
            WHERE run_date = %s;
            """,
            (run_date,),
        )
        return {str(row[0]).strip().upper() for row in cur.fetchall() if row and row[0]}


def replace_universe_map(mapping: dict[str, str]) -> None:
    if not db_enabled():
        return
    init_db()
    rows: list[tuple[str, str]] = []
    for ticker, exchange in mapping.items():
        t = str(ticker).strip().upper()
        ex = str(exchange).strip().upper()
        if not t:
            continue
        rows.append((t, ex or "UNKNOWN"))
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE production_universe_map;")
        if not rows:
            return
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO production_universe_map (
                ticker,
                exchange
            )
            VALUES %s;
            """,
            rows,
        )


def load_universe_map() -> dict[str, str]:
    if not db_enabled():
        return {}
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ticker, exchange
            FROM production_universe_map;
            """
        )
        out: dict[str, str] = {}
        for ticker, exchange in cur.fetchall():
            if not ticker:
                continue
            out[str(ticker).strip().upper()] = str(exchange or "UNKNOWN").strip().upper()
        return out


def load_excluded_tickers() -> set[str]:
    if not db_enabled():
        return set()
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ticker
            FROM production_excluded_tickers;
            """
        )
        return {
            str(row[0]).strip().upper()
            for row in cur.fetchall()
            if row and row[0] and str(row[0]).strip()
        }


def replace_excluded_tickers(tickers: Iterable[str]) -> None:
    if not db_enabled():
        return
    init_db()
    rows = []
    for ticker in tickers:
        t = str(ticker).strip().upper()
        if not t:
            continue
        rows.append((t,))
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE production_excluded_tickers;")
        if not rows:
            return
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO production_excluded_tickers (
                ticker
            )
            VALUES %s;
            """,
            rows,
        )


def _normalize_date_value(value: str) -> str:
    date_value = dt_date.fromisoformat(str(value).strip())
    return date_value.isoformat()


def upsert_run_calendar_override(
    run_date: str,
    action: str,
    reason: str | None = None,
    source: str | None = "app",
) -> None:
    if not db_enabled():
        return
    init_db()
    run_date_value = _normalize_date_value(run_date)
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"skip", "force_run"}:
        raise ValueError(
            f"Invalid run-calendar action '{action}'. Supported: ['force_run', 'skip']"
        )
    reason_value = str(reason).strip() if reason is not None else None
    source_value = str(source).strip() if source is not None else ""
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO production_run_calendar_overrides (
                run_date,
                action,
                reason,
                source
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (run_date) DO UPDATE SET
                action = EXCLUDED.action,
                reason = EXCLUDED.reason,
                source = EXCLUDED.source,
                updated_at = now();
            """,
            (run_date_value, normalized_action, reason_value, source_value),
        )


def load_run_calendar_override(run_date: str) -> dict[str, Any] | None:
    if not db_enabled():
        return None
    init_db()
    run_date_value = _normalize_date_value(run_date)
    with _connect() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT run_date::text AS run_date, action, reason, source, updated_at
            FROM production_run_calendar_overrides
            WHERE run_date = %s;
            """,
            (run_date_value,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)


def list_run_calendar_overrides(
    start: str | None = None,
    end: str | None = None,
) -> list[dict[str, Any]]:
    if not db_enabled():
        return []
    init_db()
    where: list[str] = []
    params: list[Any] = []
    if start:
        where.append("run_date >= %s")
        params.append(_normalize_date_value(start))
    if end:
        where.append("run_date <= %s")
        params.append(_normalize_date_value(end))
    where_sql = f" WHERE {' AND '.join(where)}" if where else ""
    with _connect() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"""
            SELECT run_date::text AS run_date, action, reason, source, updated_at
            FROM production_run_calendar_overrides
            {where_sql}
            ORDER BY run_date ASC;
            """,
            params,
        )
        return [dict(row) for row in cur.fetchall()]


def delete_run_calendar_override(run_date: str) -> bool:
    if not db_enabled():
        return False
    init_db()
    run_date_value = _normalize_date_value(run_date)
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM production_run_calendar_overrides WHERE run_date = %s;",
            (run_date_value,),
        )
        return cur.rowcount > 0


def upsert_state(state: ProductionState) -> None:
    if not db_enabled():
        return
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO production_state (
                id,
                last_date,
                cash,
                positions,
                prev_weights,
                total_costs_usd
            )
            VALUES (1, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                last_date = EXCLUDED.last_date,
                cash = EXCLUDED.cash,
                positions = EXCLUDED.positions,
                prev_weights = EXCLUDED.prev_weights,
                total_costs_usd = EXCLUDED.total_costs_usd,
                updated_at = now();
            """,
            (
                state.last_date,
                state.cash,
                psycopg2.extras.Json(state.positions),
                psycopg2.extras.Json(state.prev_weights),
                state.total_costs_usd,
            ),
        )


def load_state() -> ProductionState | None:
    if not db_enabled():
        return None
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT last_date, cash, positions, prev_weights, total_costs_usd
            FROM production_state
            WHERE id = 1;
            """
        )
        row = cur.fetchone()
        if not row:
            return None
        last_date, cash, positions, prev_weights, total_costs = row
        return ProductionState(
            last_date=str(last_date) if last_date else None,
            cash=float(cash or 0.0),
            positions=positions or {},
            prev_weights=prev_weights or {},
            total_costs_usd=float(total_costs or 0.0),
        )


def list_trades(limit: int, offset: int) -> tuple[int, list[dict[str, Any]]]:
    if not db_enabled():
        return 0, []
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM production_trades;")
        total = int(cur.fetchone()[0])
        cur.execute(
            """
            SELECT run_date, ticker, action, shares, price_usd, value_usd,
                   net_worth_usd, cash_usd, portfolio_value_usd, cash_weight,
                   regime, strategies
            FROM production_trades
            ORDER BY run_date DESC, ticker ASC, id DESC
            OFFSET %s LIMIT %s;
            """,
            (offset, limit if limit > 0 else total),
        )
        rows = []
        for row in cur.fetchall():
            (
                run_date,
                ticker,
                action,
                shares,
                price_usd,
                value_usd,
                net_worth_usd,
                cash_usd,
                portfolio_value_usd,
                cash_weight,
                regime,
                strategies,
            ) = row
            rows.append(
                {
                    "run_date": str(run_date) if run_date else None,
                    "date": str(run_date) if run_date else None,
                    "ticker": ticker,
                    "action": action,
                    "shares": shares,
                    "price_usd": price_usd,
                    "value_usd": value_usd,
                    "net_worth_usd": net_worth_usd,
                    "cash_usd": cash_usd,
                    "portfolio_value_usd": portfolio_value_usd,
                    "cash_weight": cash_weight,
                    "regime": regime,
                    "strategies": strategies,
                }
            )
        return total, rows


def list_run_summaries() -> list[dict[str, Any]]:
    if not db_enabled():
        return []
    init_db()
    with _connect() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT run_date AS date,
                   num_trades,
                   net_worth_usd,
                   cash_usd,
                   portfolio_value_usd,
                   cash_weight,
                   regime,
                   strategies,
                   daily_costs_usd,
                   total_costs_usd,
                   sector,
                   regime_scope,
                   cash_adjustment,
                   broker_name,
                   broker_currency,
                   broker_cash,
                   broker_cash_before,
                   broker_cash_after,
                   broker_portfolio_value,
                   broker_net_worth,
                   broker_cash_weight,
                   broker_discrepancies,
                   broker_buy_notional,
                   broker_sell_notional,
                   broker_external_flow,
                   broker_external_flow_usd,
                   broker_execution_cost,
                   broker_execution_cost_usd,
                   broker_total_execution_cost,
                   broker_total_execution_cost_usd
            FROM production_runs
            ORDER BY run_date ASC;
            """
        )
        return [dict(row) for row in cur.fetchall()]


def latest_run_date() -> str | None:
    if not db_enabled():
        return None
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT run_date FROM production_runs ORDER BY run_date DESC LIMIT 1;")
        row = cur.fetchone()
        return str(row[0]) if row and row[0] else None


def latest_summary() -> dict[str, Any] | None:
    if not db_enabled():
        return None
    init_db()
    with _connect() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT run_date AS date,
                   num_trades,
                   net_worth_usd,
                   cash_usd,
                   portfolio_value_usd,
                   cash_weight,
                   regime,
                   strategies,
                   daily_costs_usd,
                   total_costs_usd,
                   sector,
                   regime_scope,
                   cash_adjustment,
                   broker_name,
                   broker_currency,
                   broker_cash,
                   broker_cash_before,
                   broker_cash_after,
                   broker_portfolio_value,
                   broker_net_worth,
                   broker_cash_weight,
                   broker_discrepancies,
                   broker_buy_notional,
                   broker_sell_notional,
                   broker_external_flow,
                   broker_external_flow_usd,
                   broker_execution_cost,
                   broker_execution_cost_usd,
                   broker_total_execution_cost,
                   broker_total_execution_cost_usd
            FROM production_runs
            ORDER BY run_date DESC
            LIMIT 1;
            """
        )
        row = cur.fetchone()
        return dict(row) if row else None


def latest_trades(limit: int = 0) -> list[dict[str, Any]]:
    if not db_enabled():
        return []
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT run_date FROM production_runs ORDER BY run_date DESC LIMIT 1;")
        row = cur.fetchone()
        if not row or not row[0]:
            return []
        run_date = row[0]
        cur.execute(
            """
            SELECT run_date, ticker, action, shares, price_usd, value_usd,
                   net_worth_usd, cash_usd, portfolio_value_usd, cash_weight,
                   regime, strategies
            FROM production_trades
            WHERE run_date = %s
            ORDER BY ticker ASC, id DESC;
            """,
            (run_date,),
        )
        rows = []
        for row in cur.fetchall():
            (
                run_date,
                ticker,
                action,
                shares,
                price_usd,
                value_usd,
                net_worth_usd,
                cash_usd,
                portfolio_value_usd,
                cash_weight,
                regime,
                strategies,
            ) = row
            rows.append(
                {
                    "run_date": str(run_date) if run_date else None,
                    "date": str(run_date) if run_date else None,
                    "ticker": ticker,
                    "action": action,
                    "shares": shares,
                    "price_usd": price_usd,
                    "value_usd": value_usd,
                    "net_worth_usd": net_worth_usd,
                    "cash_usd": cash_usd,
                    "portfolio_value_usd": portfolio_value_usd,
                    "cash_weight": cash_weight,
                    "regime": regime,
                    "strategies": strategies,
                }
            )
        if limit and limit > 0:
            return rows[:limit]
        return rows


def upsert_broker_account(run_date: str, broker: str, summary: dict[str, Any]) -> None:
    if not db_enabled():
        return
    init_db()
    currency = summary.get("currency")
    cash = summary.get("cash")
    investments = summary.get("investments")
    net_worth = summary.get("net_worth")
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO production_broker_account (
                run_date,
                broker,
                currency,
                cash,
                investments,
                net_worth,
                payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_date, broker) DO UPDATE SET
                currency = EXCLUDED.currency,
                cash = EXCLUDED.cash,
                investments = EXCLUDED.investments,
                net_worth = EXCLUDED.net_worth,
                payload = EXCLUDED.payload;
            """,
            (
                run_date,
                broker,
                currency,
                cash,
                investments,
                net_worth,
                psycopg2.extras.Json(summary.get("payload")) if summary.get("payload") is not None else None,
            ),
        )


def replace_broker_positions(
    run_date: str,
    broker: str,
    positions: Iterable[dict[str, Any]],
) -> None:
    if not db_enabled():
        return
    init_db()
    rows = []
    for pos in positions:
        rows.append(
            (
                run_date,
                broker,
                pos.get("ticker"),
                pos.get("quantity"),
                pos.get("average_price"),
                pos.get("current_price"),
                pos.get("instrument_currency"),
                pos.get("account_currency"),
                pos.get("wallet_current_value"),
                psycopg2.extras.Json(pos.get("payload")) if pos.get("payload") is not None else None,
            )
        )
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM production_broker_positions WHERE run_date = %s AND broker = %s;",
            (run_date, broker),
        )
        if not rows:
            return
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO production_broker_positions (
                run_date,
                broker,
                ticker,
                quantity,
                average_price,
                current_price,
                instrument_currency,
                account_currency,
                wallet_current_value,
                payload
            )
            VALUES %s;
            """,
            rows,
        )


def replace_broker_orders(
    run_date: str,
    broker: str,
    orders: Iterable[dict[str, Any]],
) -> None:
    if not db_enabled():
        return
    init_db()
    rows = []
    for order in orders:
        rows.append(
            (
                run_date,
                broker,
                order.get("ticker"),
                order.get("action"),
                order.get("quantity"),
                order.get("filled_quantity"),
                order.get("exec_price"),
                order.get("currency"),
                order.get("status"),
                order.get("order_id"),
                psycopg2.extras.Json(order.get("payload")) if order.get("payload") is not None else None,
            )
        )
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM production_broker_orders WHERE run_date = %s AND broker = %s;",
            (run_date, broker),
        )
        if not rows:
            return
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO production_broker_orders (
                run_date,
                broker,
                ticker,
                action,
                quantity,
                filled_quantity,
                exec_price,
                currency,
                status,
                order_id,
                payload
            )
            VALUES %s;
            """,
            rows,
        )


def latest_broker_account(broker: str) -> dict[str, Any] | None:
    if not db_enabled():
        return None
    init_db()
    with _connect() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT run_date,
                   broker,
                   currency,
                   cash,
                   investments,
                   net_worth,
                   payload
            FROM production_broker_account
            WHERE broker = %s
            ORDER BY run_date DESC
            LIMIT 1;
            """,
            (broker,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def latest_broker_positions(broker: str) -> list[dict[str, Any]]:
    if not db_enabled():
        return []
    init_db()
    with _connect() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT run_date,
                   broker,
                   ticker,
                   quantity,
                   average_price,
                   current_price,
                   instrument_currency,
                   account_currency,
                   wallet_current_value,
                   payload
            FROM production_broker_positions
            WHERE broker = %s
            AND run_date = (
                SELECT MAX(run_date) FROM production_broker_positions WHERE broker = %s
            )
            ORDER BY ticker ASC;
            """,
            (broker, broker),
        )
        return [dict(row) for row in cur.fetchall()]


def list_broker_orders(broker: str, limit: int, offset: int) -> tuple[int, list[dict[str, Any]]]:
    if not db_enabled():
        return 0, []
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM production_broker_orders WHERE broker = %s;",
            (broker,),
        )
        total = int(cur.fetchone()[0] or 0)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT run_date,
                   broker,
                   ticker,
                   action,
                   quantity,
                   filled_quantity,
                   exec_price,
                   currency,
                   status,
                   order_id,
                   payload,
                   created_at
            FROM production_broker_orders
            WHERE broker = %s
            ORDER BY run_date DESC, id DESC
            LIMIT %s OFFSET %s;
            """,
            (broker, limit, offset),
        )
        return total, [dict(row) for row in cur.fetchall()]


def latest_broker_orders(broker: str, limit: int = 0) -> list[dict[str, Any]]:
    if not db_enabled():
        return []
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT run_date FROM production_broker_orders WHERE broker = %s ORDER BY run_date DESC LIMIT 1;",
            (broker,),
        )
        row = cur.fetchone()
        if not row or not row[0]:
            return []
        run_date = row[0]
        query = """
            SELECT run_date,
                   broker,
                   ticker,
                   action,
                   quantity,
                   filled_quantity,
                   exec_price,
                   currency,
                   status,
                   order_id,
                   payload,
                   created_at
            FROM production_broker_orders
            WHERE broker = %s AND run_date = %s
            ORDER BY id DESC
        """
        params: tuple[Any, ...] = (broker, run_date)
        if limit and limit > 0:
            query += " LIMIT %s"
            params = (broker, run_date, limit)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]


def replace_universe_monitor_snapshot(
    summary: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
) -> None:
    if not db_enabled():
        return
    init_db()
    run_date = summary.get("run_date")
    if not run_date:
        return
    generated_at = summary.get("generated_at_utc")

    rows: list[tuple[Any, ...]] = []
    for item in candidates:
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        rows.append(
            (
                ticker,
                run_date,
                item.get("exchange"),
                item.get("symbol"),
                item.get("in_current_universe"),
                item.get("tv_data"),
                item.get("tv_pass"),
                item.get("tv_fail_reasons"),
                item.get("sector"),
                item.get("tech_sector"),
                item.get("quality_pass"),
                item.get("quality_reasons"),
                item.get("monitor_pass"),
                item.get("pass_streak"),
                item.get("total_pass_days"),
                item.get("close"),
                item.get("volume"),
                item.get("dollar_volume"),
                item.get("market_cap_basic"),
                item.get("recommend_all"),
                item.get("sma50"),
                item.get("sma200"),
                item.get("trend_up"),
                item.get("quality_rows"),
                item.get("quality_median_adv_dollars"),
                item.get("quality_p20_adv_dollars"),
                item.get("quality_p05_price"),
                item.get("quality_median_vol21"),
                item.get("quality_p95_abs_log_return"),
                item.get("metadata_fail_reasons"),
                item.get("market_cap_pass"),
                item.get("yfinance_market_cap"),
            )
        )

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE production_universe_monitor_candidates;")
        if rows:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO production_universe_monitor_candidates (
                    ticker,
                    run_date,
                    exchange,
                    symbol,
                    in_current_universe,
                    tv_data,
                    tv_pass,
                    tv_fail_reasons,
                    sector,
                    tech_sector,
                    quality_pass,
                    quality_reasons,
                    monitor_pass,
                    pass_streak,
                    total_pass_days,
                    close,
                    volume,
                    dollar_volume,
                    market_cap_basic,
                    recommend_all,
                    sma50,
                    sma200,
                    trend_up,
                    quality_rows,
                    quality_median_adv_dollars,
                    quality_p20_adv_dollars,
                    quality_p05_price,
                    quality_median_vol21,
                    quality_p95_abs_log_return,
                    metadata_fail_reasons,
                    market_cap_pass,
                    yfinance_market_cap
                )
                VALUES %s;
                """,
                rows,
            )

        cur.execute(
            """
            INSERT INTO production_universe_monitor_state (
                id,
                run_date,
                generated_at_utc,
                candidates_evaluated,
                tv_pass_count,
                tech_tv_pass_count,
                market_cap_pass_count,
                tech_after_market_cap_count,
                quality_pass_count,
                potential_additions_count,
                min_pass_days,
                payload
            )
            VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                run_date = EXCLUDED.run_date,
                generated_at_utc = EXCLUDED.generated_at_utc,
                candidates_evaluated = EXCLUDED.candidates_evaluated,
                tv_pass_count = EXCLUDED.tv_pass_count,
                tech_tv_pass_count = EXCLUDED.tech_tv_pass_count,
                market_cap_pass_count = EXCLUDED.market_cap_pass_count,
                tech_after_market_cap_count = EXCLUDED.tech_after_market_cap_count,
                quality_pass_count = EXCLUDED.quality_pass_count,
                potential_additions_count = EXCLUDED.potential_additions_count,
                min_pass_days = EXCLUDED.min_pass_days,
                payload = EXCLUDED.payload,
                updated_at = now();
            """,
            (
                run_date,
                generated_at,
                summary.get("candidates_evaluated"),
                summary.get("tv_pass_count"),
                summary.get("tech_tv_pass_count"),
                summary.get("market_cap_pass_count"),
                summary.get("tech_after_market_cap_count"),
                summary.get("quality_pass_count"),
                summary.get("potential_additions_count"),
                summary.get("min_pass_days"),
                psycopg2.extras.Json(summary),
            ),
        )


def latest_universe_monitor_summary() -> dict[str, Any] | None:
    if not db_enabled():
        return None
    init_db()
    with _connect() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT run_date,
                   generated_at_utc,
                   candidates_evaluated,
                   tv_pass_count,
                   tech_tv_pass_count,
                   market_cap_pass_count,
                   tech_after_market_cap_count,
                   quality_pass_count,
                   potential_additions_count,
                   min_pass_days,
                   payload
            FROM production_universe_monitor_state
            WHERE id = 1;
            """
        )
        row = cur.fetchone()
        return dict(row) if row else None


def list_universe_monitor_candidates(
    limit: int,
    offset: int,
    watchlist: bool = False,
    potential: bool = False,
) -> tuple[int, list[dict[str, Any]]]:
    if not db_enabled():
        return 0, []
    init_db()

    min_pass_days = 0
    if potential:
        state = latest_universe_monitor_summary()
        if state:
            min_pass_days = int(state.get("min_pass_days") or 0)

    where: list[str] = []
    params: list[Any] = []
    if watchlist:
        where.append("tv_pass = TRUE AND tech_sector = TRUE AND market_cap_pass = TRUE")
    if potential:
        where.append("monitor_pass = TRUE AND pass_streak >= %s")
        params.append(min_pass_days)

    where_sql = ""
    if where:
        where_sql = " WHERE " + " AND ".join(where)

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT COUNT(*) FROM production_universe_monitor_candidates{where_sql};",
            params,
        )
        total = int(cur.fetchone()[0] or 0)

        query = f"""
            SELECT ticker,
                   run_date,
                   exchange,
                   symbol,
                   in_current_universe,
                   tv_data,
                   tv_pass,
                   tv_fail_reasons,
                   sector,
                   tech_sector,
                   quality_pass,
                   quality_reasons,
                   monitor_pass,
                   pass_streak,
                   total_pass_days,
                   close,
                   volume,
                   dollar_volume,
                   market_cap_basic,
                   recommend_all,
                   sma50,
                   sma200,
                   trend_up,
                   quality_rows,
                   quality_median_adv_dollars,
                   quality_p20_adv_dollars,
                   quality_p05_price,
                   quality_median_vol21,
                   quality_p95_abs_log_return,
                   metadata_fail_reasons,
                   market_cap_pass,
                   yfinance_market_cap
            FROM production_universe_monitor_candidates
            {where_sql}
            ORDER BY pass_streak DESC,
                     monitor_pass DESC,
                     recommend_all DESC NULLS LAST,
                     dollar_volume DESC NULLS LAST,
                     ticker ASC
            OFFSET %s LIMIT %s;
        """
        page_params = [*params, offset, limit if limit > 0 else total]
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(query, page_params)
        return total, [dict(row) for row in cur.fetchall()]


def append_pending_adjustments(entries: Iterable[dict[str, Any]]) -> None:
    if not db_enabled():
        return
    init_db()
    rows = [(psycopg2.extras.Json(entry),) for entry in entries]
    if not rows:
        return
    with _connect() as conn:
        cur = conn.cursor()
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO production_pending_adjustments (payload) VALUES %s;",
            rows,
        )


def load_pending_adjustments() -> list[dict[str, Any]]:
    if not db_enabled():
        return []
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT payload FROM production_pending_adjustments ORDER BY id ASC;"
        )
        return [row[0] for row in cur.fetchall() if row and row[0]]


def clear_pending_adjustments() -> None:
    if not db_enabled():
        return
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM production_pending_adjustments;")


def reset_production_data(preserve_universe_monitor: bool = True) -> None:
    if not db_enabled():
        raise RuntimeError("DATABASE_URL/POSTGRES_URL not configured.")
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        tables = [
            "production_runs",
            "production_trades",
            "production_prices",
            "production_state",
            "production_pending_adjustments",
            "production_broker_account",
            "production_broker_positions",
            "production_broker_orders",
        ]
        if not preserve_universe_monitor:
            tables.extend(
                [
                    "production_universe_monitor_state",
                    "production_universe_monitor_candidates",
                ]
            )
        cur.execute(f"TRUNCATE {', '.join(tables)};")
