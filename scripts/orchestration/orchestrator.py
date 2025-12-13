import os
import sys
import json
import subprocess
import shutil
import time
import pandas as pd
import inspect
import hashlib

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src import config
from src.llm_interface import get_new_alpha_idea, clean_llm_code
from src.walkforward import run_walk_forward

# --- CONFIGURATION ---
MAX_ITERATIONS = 10
ALPHAS_DIR = os.path.join(PROJECT_ROOT, "alphas")
CANDIDATES_DIR = os.path.join(PROJECT_ROOT, "_candidates")
ENSEMBLE_HISTORY_FILE = os.path.join(PROJECT_ROOT, "ensemble_history.json")
PYTHON_EXEC = sys.executable
BACKTESTER_SCRIPT = os.path.join(PROJECT_ROOT, "scripts/backtesting/backtester.py")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts/training/train_alpha.py")

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
    """Gets a summary of existing alphas."""
    if not os.path.exists(alphas_dir):
        return "No alphas exist yet.", []
    alpha_names = [d for d in os.listdir(alphas_dir) if os.path.isdir(os.path.join(alphas_dir, d)) and not d.startswith("_")]
    summary_lines = []
    for name in alpha_names:
        desc_path = os.path.join(alphas_dir, name, "description.txt")
        if os.path.exists(desc_path):
            with open(desc_path, 'r') as f:
                description = f.read().strip()
                summary_lines.append(f"- {name}: {description}")
    return "\n".join(summary_lines) if summary_lines else "No alphas exist yet.", alpha_names

def run_backtest(alpha_list):
    """Runs the backtester for a given list of alphas and returns the CAGR."""
    if not alpha_list:
        print("No alphas to backtest. Returning baseline CAGR of -inf.")
        return -float('inf')

    cmd = [PYTHON_EXEC, BACKTESTER_SCRIPT, "--alphas"] + alpha_list
    print(f"\n--- Running backtest for: {alpha_list} ---")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
        print(result.stdout)
        ensemble_name = make_ensemble_dirname(alpha_list)
        results_candidates = [
            os.path.join(ALPHAS_DIR, "_ensembles", ensemble_name, "results.json")
        ]
        if len(alpha_list) == 1:
            results_candidates.append(os.path.join(ALPHAS_DIR, ensemble_name, "results.json"))
        else:
            legacy_name = "_vs_".join(sorted(alpha_list))
            results_candidates.append(os.path.join(ALPHAS_DIR, "_ensembles", legacy_name, "results.json"))

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

def main():
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
        baseline_cagr = run_backtest(existing_alpha_names)
        print(f"Baseline Ensemble CAGR: {baseline_cagr:.2%}")

        failed_attempts_summary = "\n".join([f"- {item['summary']}" for item in history])
        if not failed_attempts_summary:
            failed_attempts_summary = "No failed attempts yet."
        
        print("\n--- Consulting LLM for new alpha strategy ---")
        try:
            prompt, new_script_raw = get_new_alpha_idea(alpha_summary, baseline_cagr, failed_attempts_summary, data_schema)
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
            description = script_globals['DESCRIPTION']
        except Exception as e:
            print(f"Could not parse LLM response for description: {e}")
            description = "LLM response did not contain a valid DESCRIPTION variable."

        print(f"LLM proposed change: {description}")

        candidate_name = f"candidate_{int(time.time())}"
        candidate_dir = os.path.join(CANDIDATES_DIR, candidate_name)
        os.makedirs(candidate_dir, exist_ok=True)

        with open(os.path.join(candidate_dir, "feature_engineering.py"), "w") as f:
            f.write(new_script)
        with open(os.path.join(candidate_dir, "description.txt"), "w") as f:
            f.write(description)

        print(f"\n--- Training candidate alpha: {candidate_name} ---")
        # The output of the training is the alpha directory, which is inside the candidate_dir
        output_alpha_dir = os.path.join(candidate_dir, candidate_name)
        train_cmd = [PYTHON_EXEC, TRAIN_SCRIPT, "--alpha-name", candidate_name, "--description", description, "--output-dir", candidate_dir]
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = PROJECT_ROOT
            # We need to pass the feature engineering file to the training script
            train_cmd.extend(["--features-file", os.path.join(candidate_dir, "feature_engineering.py")])
            
            # Stream the output by not capturing it, allowing the user to see the progress
            print("Training progress will be displayed below:")
            subprocess.run(train_cmd, check=True, env=env, cwd=PROJECT_ROOT)

        except subprocess.CalledProcessError as e:
            print(f"Candidate training failed. Review the output above. Error: {e}")
            shutil.rmtree(candidate_dir)
            continue

        print("\n--- Evaluating candidate's contribution to the ensemble ---")
        # Move the trained alpha to the main alphas directory
        shutil.move(output_alpha_dir, os.path.join(ALPHAS_DIR, candidate_name))
        
        new_cagr = -float('inf')
        if config.ENABLE_WALK_FORWARD_VALIDATION:
            print(f"--- Evaluating candidate via Walk-Forward Validation: {candidate_name} ---")
            metrics = run_walk_forward(candidate_name)
            if metrics and 'CAGR' in metrics:
                # run_walk_forward returns CAGR in percent; convert to decimal for fair comparison
                new_cagr = metrics['CAGR'] / 100.0
                print(f"Walk-Forward CAGR for {candidate_name}: {new_cagr:.2%}")
            else:
                print(f"Walk-forward validation failed for {candidate_name}.")
        else:
            print(f"--- Evaluating candidate via Simple Backtest: {candidate_name} ---")
            new_ensemble_list = existing_alpha_names + [candidate_name]
            new_cagr = run_backtest(new_ensemble_list)
            print(f"New Ensemble CAGR (with candidate): {new_cagr:.2%}")

        if new_cagr > baseline_cagr:
            print(f"\n--- ✅ SUCCESS: Candidate {candidate_name} improved performance. Promoting. ---")
            # The candidate dir is now empty, except for the feature_engineering.py and description.txt
            # The trained model is in the ALPHAS_DIR. We can keep the candidate dir for reference
            pass
        else:
            print(f"\n--- 😔 FAILURE: Candidate {candidate_name} did not improve performance. Keeping artifacts for inspection. ---")
            history.append({"summary": description, "code": new_script})
            with open(ENSEMBLE_HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
            # Leave alphas/candidate and candidate_dir for debugging
        
        # Do not delete candidate_dir; keep for inspection

    print(f"\n{'='*30} Orchestration Finished {'='*30}")

if __name__ == "__main__":
    main()
