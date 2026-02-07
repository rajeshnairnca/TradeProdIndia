from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_smoke_data(path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=16, freq="D")
    tickers = ["AAA", "BBB"]
    rows = []
    for i, dt in enumerate(dates):
        rows.append((dt, "AAA", 100.0 + i, 0.02, 2_000_000.0))
        rows.append((dt, "BBB", 100.0 + (i * 0.2), 0.03, 2_000_000.0))
    df = pd.DataFrame(rows, columns=["date", "ticker", "Close", "vol_21", "adv_21"])
    df = df.set_index(["date", "ticker"]).sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _write_strategy(strategy_root: Path) -> None:
    strategy_dir = strategy_root / "smoke_alpha"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    strategy_file = strategy_dir / "strategy.py"
    strategy_file.write_text(
        "\n".join(
            [
                "DESCRIPTION = 'smoke strategy'",
                "REGIME_TAGS = []",
                "",
                "import pandas as pd",
                "",
                "def generate_scores(df):",
                "    rank = {'AAA': 2.0, 'BBB': 1.0}",
                "    values = df.index.get_level_values('ticker').map(rank).astype(float)",
                "    return pd.Series(values, index=df.index, dtype=float)",
                "",
            ]
        )
    )


def test_backtester_cli_smoke(tmp_path: Path):
    repo_root = _repo_root()
    data_path = tmp_path / "daily_data_smoke.parquet"
    strategy_root = tmp_path / "alphas"
    output_root = tmp_path / "out"
    _build_smoke_data(data_path)
    _write_strategy(strategy_root)

    env = dict(os.environ)
    env["DATA_FILE"] = str(data_path)
    env["TRADING_REGION"] = "us"
    env["MPLBACKEND"] = "Agg"

    cmd = [
        sys.executable,
        "scripts/backtesting/backtester.py",
        "--strategies",
        "smoke_alpha",
        "--strategy-roots",
        str(strategy_root),
        "--output-root",
        str(output_root),
        "--use-full-history",
    ]
    subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    results_path = output_root / "_ensembles" / "us" / "smoke_alpha" / "results.json"
    assert results_path.exists()
    payload = json.loads(results_path.read_text())
    assert payload["num_strategies"] == 1
    assert payload["strategies"] == ["smoke_alpha"]
