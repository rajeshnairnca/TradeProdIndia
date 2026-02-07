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
                cash_adjustment double precision,
                broker_name text,
                broker_currency text,
                broker_cash double precision,
                broker_portfolio_value double precision,
                broker_net_worth double precision,
                broker_cash_weight double precision,
                broker_discrepancies jsonb
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
                broker_portfolio_value,
                broker_net_worth,
                broker_cash_weight,
                broker_discrepancies
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                broker_portfolio_value = EXCLUDED.broker_portfolio_value,
                broker_net_worth = EXCLUDED.broker_net_worth,
                broker_cash_weight = EXCLUDED.broker_cash_weight,
                broker_discrepancies = EXCLUDED.broker_discrepancies;
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
                summary.get("broker_portfolio_value"),
                summary.get("broker_net_worth"),
                summary.get("broker_cash_weight"),
                psycopg2.extras.Json(summary.get("broker_discrepancies"))
                if summary.get("broker_discrepancies") is not None
                else None,
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
                   broker_portfolio_value,
                   broker_net_worth,
                   broker_cash_weight,
                   broker_discrepancies
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
                   broker_portfolio_value,
                   broker_net_worth,
                   broker_cash_weight,
                   broker_discrepancies
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


def reset_production_data() -> None:
    if not db_enabled():
        raise RuntimeError("DATABASE_URL/POSTGRES_URL not configured.")
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            TRUNCATE
                production_runs,
                production_trades,
                production_prices,
                production_state,
                production_pending_adjustments,
                production_broker_account,
                production_broker_positions,
                production_broker_orders;
            """
        )
