import os
import sys
import json
import subprocess
import shutil
import time
import pandas as pd
import argparse
import hashlib
import re
import numpy as np

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.llm_interface import get_new_alpha_idea, clean_llm_code
from src.strategy import load_strategies
from src.strategy_sweep import sweep_strategy_combinations

# --- CONFIGURATION ---
MAX_ITERATIONS = 10
ALPHAS_ROOT = os.getenv("ALPHAS_ROOT", "alphas")
CANDIDATES_ROOT = os.getenv("CANDIDATES_ROOT", "_candidates")
ALPHAS_DIR = os.path.join(PROJECT_ROOT, ALPHAS_ROOT)
CANDIDATES_DIR = os.path.join(PROJECT_ROOT, CANDIDATES_ROOT)
ENSEMBLE_HISTORY_FILE = os.path.join(
    PROJECT_ROOT,
    f"ensemble_history_{config.TRADING_REGION}.json",
)
# TODO: Consider adding a supervised triple-barrier pathway (profit/stop/time labels) to generate swing-friendly signals, then feed those as features or signals into the allocator.
PYTHON_EXEC = sys.executable
BACKTESTER_SCRIPT = os.path.join(PROJECT_ROOT, "scripts/backtesting/backtester.py")
STRATEGY_FILE = "strategy.py"

def make_ensemble_dirname(alpha_names: list[str]) -> str:
    """Generate a short, stable ensemble folder name with a hash suffix."""
    sorted_names = sorted(alpha_names)
    if len(sorted_names) == 1:
        return sorted_names[0]

    digest = hashlib.sha1("::".join(sorted_names).encode("utf-8")).hexdigest()[:8]
    preview_parts = [name[:12] for name in sorted_names[:3]]
    preview = "_vs_".join(preview_parts)
    if len(sorted_names) > 3:
        preview += f"_plus{len(sorted_names) - 3}"
    return f"{preview}__{digest}"

def get_existing_alphas(alphas_dir):
    """Gets a summary of existing rule-based strategies."""
    if not os.path.exists(alphas_dir):
        return "No alphas exist yet.", []
    alpha_names = [
        d
        for d in os.listdir(alphas_dir)
        if os.path.isdir(os.path.join(alphas_dir, d))
        and not d.startswith("_")
        and os.path.exists(os.path.join(alphas_dir, d, STRATEGY_FILE))
    ]
    summary_lines = []
    for name in alpha_names:
        desc_path = os.path.join(alphas_dir, name, "description.txt")
        if os.path.exists(desc_path):
            with open(desc_path, 'r') as f:
                description = f.read().strip()
                summary_lines.append(f"- {name}: {description}")
    return "\n".join(summary_lines) if summary_lines else "No alphas exist yet.", alpha_names

def run_backtest(alpha_list, strategy_roots=None):
    """Runs the backtester for a given list of strategies and returns the CAGR."""
    if not alpha_list:
        print("No strategies to backtest. Returning baseline CAGR of -inf.")
        return -float("inf")

    cmd = [PYTHON_EXEC, BACKTESTER_SCRIPT, "--strategies"] + alpha_list
    roots = strategy_roots or [ALPHAS_ROOT]
    for root in roots:
        cmd.extend(["--strategy-roots", root])
    print(f"\n--- Running backtest for: {alpha_list} ---")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
        print(result.stdout)
        ensemble_name = make_ensemble_dirname(alpha_list)
        region = config.TRADING_REGION
        results_candidates = [os.path.join(ALPHAS_DIR, "_ensembles", region, ensemble_name, "results.json")]

        for results_path in results_candidates:
            if not os.path.exists(results_path):
                continue
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    return results.get("cagr", -float('inf'))
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {results_path}.")
                continue
        print(f"Backtest finished but results.json not found in: {results_candidates}")
        return -float('inf')
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Backtest failed for {alpha_list}. Error: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"Stderr: {e.stderr}")
        return -float('inf')


def validate_llm_strategy(script_text: str, sample_df: pd.DataFrame) -> tuple[bool, str]:
    lookahead_patterns = [
        r"center\s*=\s*True",
        r"shift\s*\(\s*-\s*\d",
        r"\blead\s*\(",
        r"\bshift\s*\(\s*-\s*1",
    ]
    for pat in lookahead_patterns:
        if re.search(pat, script_text):
            return False, f"Lookahead-like pattern detected: {pat}"

    script_globals = {}
    try:
        exec(script_text, script_globals)
    except Exception as e:
        return False, f"Script execution failed: {e}"

    score_fn = script_globals.get("generate_scores")
    if not callable(score_fn):
        return False, "Missing generate_scores(df) function."

    try:
        scores = score_fn(sample_df)
    except Exception as e:
        return False, f"generate_scores failed on sample data: {e}"

    if not isinstance(scores, pd.Series):
        return False, "generate_scores must return a pandas Series."
    if len(scores) != len(sample_df):
        return False, "generate_scores returned wrong length."
    if not scores.index.equals(sample_df.index):
        return False, "generate_scores must return scores indexed like the input DataFrame."
    numeric_scores = pd.to_numeric(scores, errors="coerce")
    if numeric_scores.isna().all():
        return False, "generate_scores returned all-NaN or non-numeric values."
    if not np.isfinite(numeric_scores.fillna(0.0)).all():
        return False, "generate_scores returned infinite values."

    return True, "ok"

def main():
    parser = argparse.ArgumentParser(description="Orchestrate alpha generation/training/backtesting.")
    parser.add_argument("--llm-guidance", help="Optional extra guidance to pass to the LLM when proposing new alphas.")
    parser.add_argument("--sweep-baseline", action="store_true", help="Use a combination sweep to pick the best baseline ensemble.")
    parser.add_argument("--sweep-min-size", type=int, default=1, help="Minimum strategy count for sweep combos.")
    parser.add_argument("--sweep-max-size", type=int, default=None, help="Maximum strategy count for sweep combos.")
    parser.add_argument("--sweep-max-combos", type=int, default=200, help="Maximum combinations to evaluate per sweep.")
    parser.add_argument("--sweep-metric", default="cagr", choices=["cagr", "sharpe", "max_drawdown"], help="Metric to optimize during sweeps.")
    args = parser.parse_args()

    llm_guidance = args.llm_guidance or os.getenv("LLM_GUIDANCE", "")
    os.makedirs(ALPHAS_DIR, exist_ok=True)
    os.makedirs(CANDIDATES_DIR, exist_ok=True)

    if os.path.exists(ENSEMBLE_HISTORY_FILE):
        with open(ENSEMBLE_HISTORY_FILE, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = [] # Treat empty or invalid json as an empty history
    else:
        history = []

    # Get the data schema to pass to the LLM
    try:
        df = pd.read_parquet(os.path.join(PROJECT_ROOT, config.DATA_FILE))
        data_schema = str(list(df.columns))
    except Exception as e:
        print(f"Fatal: Could not read data schema from {config.DATA_FILE}. Error: {e}")
        return

    for i in range(MAX_ITERATIONS):
        print(f"\n{'='*30} ORCHESTRATOR ITERATION {i+1}/{MAX_ITERATIONS} {'='*30}")

        alpha_summary, existing_alpha_names = get_existing_alphas(ALPHAS_DIR)
        print("Current Ensemble:")
        print(alpha_summary)
        baseline_combo = existing_alpha_names
        if args.sweep_baseline and existing_alpha_names:
            base_strategies = load_strategies(existing_alpha_names, [ALPHAS_ROOT])
            strategies_by_name = {s.name: s for s in base_strategies}
            sweep = sweep_strategy_combinations(
                df=df,
                strategies_by_name=strategies_by_name,
                min_size=args.sweep_min_size,
                max_size=args.sweep_max_size,
                max_combos=args.sweep_max_combos,
                seed=config.SEED,
                metric=args.sweep_metric,
            )
            baseline_combo = list(sweep.best_combo)
            baseline_cagr = sweep.best_metrics.get("CAGR", 0.0) / 100.0
            print(f"Best baseline combo: {baseline_combo}")
        else:
            baseline_cagr = run_backtest(existing_alpha_names, strategy_roots=[ALPHAS_ROOT])
        print(f"Baseline Ensemble CAGR: {baseline_cagr:.2%}")

        failed_attempts_summary = "\n".join([f"- {item['summary']}" for item in history])
        if not failed_attempts_summary:
            failed_attempts_summary = "No failed attempts yet."
        
        print("\n--- Consulting LLM for new alpha strategy ---")
        try:
            prompt, new_script_raw = get_new_alpha_idea(alpha_summary, baseline_cagr, failed_attempts_summary, data_schema, guidance=llm_guidance)
            new_script = clean_llm_code(new_script_raw)
        except Exception as e:
            print(f"LLM call failed (network or auth issue): {e}")
            break
        
        if not new_script:
            print("LLM did not return a valid script. Skipping iteration.")
            continue

        try:
            script_globals = {}
            exec(new_script, script_globals)
            description = script_globals.get("DESCRIPTION", "").strip() or "LLM response did not contain a valid DESCRIPTION."
            if not callable(script_globals.get("generate_scores")):
                print("LLM response missing generate_scores(df). Skipping iteration.")
                continue
            if not script_globals.get("REGIME_TAGS"):
                print("LLM response missing REGIME_TAGS. Skipping iteration.")
                continue
        except Exception as e:
            print(f"Could not parse LLM response: {e}")
            continue

        sample_df = df.dropna(subset=["Close"]).tail(5000)
        if sample_df.empty:
            sample_df = df.tail(5000)
        ok, reason = validate_llm_strategy(new_script, sample_df)
        if not ok:
            print(f"LLM strategy validation failed: {reason}")
            continue

        print(f"LLM proposed change: {description}")

        candidate_name = f"candidate_{int(time.time())}"
        candidate_dir = os.path.join(CANDIDATES_DIR, candidate_name)
        os.makedirs(candidate_dir, exist_ok=True)

        with open(os.path.join(candidate_dir, STRATEGY_FILE), "w") as f:
            f.write(new_script)
        with open(os.path.join(candidate_dir, "description.txt"), "w") as f:
            f.write(description)

        print("\n--- Evaluating candidate's contribution to the ensemble ---")
        new_cagr = -float('inf')
        if args.sweep_baseline and existing_alpha_names:
            print(f"--- Evaluating candidate via sweep: {candidate_name} ---")
            candidate_list = existing_alpha_names + [candidate_name]
            candidate_strategies = load_strategies(candidate_list, [ALPHAS_ROOT, CANDIDATES_ROOT])
            strategies_by_name = {s.name: s for s in candidate_strategies}
            sweep = sweep_strategy_combinations(
                df=df,
                strategies_by_name=strategies_by_name,
                min_size=args.sweep_min_size,
                max_size=args.sweep_max_size,
                max_combos=args.sweep_max_combos,
                seed=config.SEED,
                metric=args.sweep_metric,
            )
            new_cagr = sweep.best_metrics.get("CAGR", 0.0) / 100.0
            print(f"Best combo with candidate: {list(sweep.best_combo)}")
            print(f"Sweep CAGR for ensemble with {candidate_name}: {new_cagr:.2%}")
        else:
            print(f"--- Evaluating candidate via Simple Backtest: {candidate_name} ---")
            new_ensemble_list = existing_alpha_names + [candidate_name]
            new_cagr = run_backtest(new_ensemble_list, strategy_roots=[ALPHAS_ROOT, CANDIDATES_ROOT])
            print(f"New Ensemble CAGR (with candidate): {new_cagr:.2%}")

        if new_cagr > baseline_cagr:
            print(f"\n--- ✅ SUCCESS: Candidate {candidate_name} improved performance. Promoting. ---")
            shutil.move(candidate_dir, os.path.join(ALPHAS_DIR, candidate_name))
        else:
            print(f"\n--- 😔 FAILURE: Candidate {candidate_name} did not improve performance. Keeping artifacts for inspection. ---")
            history.append({"summary": description, "code": new_script})
            with open(ENSEMBLE_HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
            # Leave candidate_dir for debugging
        
        # Do not delete candidate_dir; keep for inspection

    print(f"\n{'='*30} Orchestration Finished {'='*30}")

if __name__ == "__main__":
    main()
