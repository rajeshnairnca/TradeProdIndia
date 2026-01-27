from __future__ import annotations

import os
from contextlib import contextmanager
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
    conn.autocommit = True
    try:
        yield conn
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
                cash_adjustment double precision
            );
            """
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
            "CREATE INDEX IF NOT EXISTS production_trades_run_date_idx ON production_trades (run_date);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS production_trades_ticker_idx ON production_trades (ticker);"
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
                cash_adjustment
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                cash_adjustment = EXCLUDED.cash_adjustment;
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
                   cash_adjustment
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
                   cash_adjustment
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
