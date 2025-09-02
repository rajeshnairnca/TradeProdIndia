# tune_hyperparameters.py

import os
import sys
import subprocess
import json
import re
import uuid
import optuna
from datetime import datetime

# --- Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# --- Configuration ---
AGENT_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "agent_v0.py")
DOCKER_IMAGE_NAME = "rl-agent-env"
N_TRIALS = 50
TUNING_LOG_FILE = os.path.join(PROJECT_ROOT, "optuna_tuning_log.txt")

def objective(trial: optuna.Trial) -> float:
    params = {
        "TOP_K": trial.suggest_int("TOP_K", 5, 15),
        "TURNOVER_PENALTY": trial.suggest_float("TURNOVER_PENALTY", 1e-5, 1e-3, log=True),
        "RISK_PENALTY_COEFF": trial.suggest_float("RISK_PENALTY_COEFF", 0.0, 1.0),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05),
    }

    with open(AGENT_TEMPLATE_PATH, 'r') as f: agent_code = f.read()
    agent_code = re.sub(r"(\bTOP_K\b\s*=\s*)\\d+", fr"\g<1>{params['TOP_K']}", agent_code)
    agent_code = re.sub(r"(\bTURNOVER_PENALTY\b\s*=\s*)[0-9e.-]+", fr"\g<1>{params['TURNOVER_PENALTY']}", agent_code)
    agent_code = re.sub(r"(\bRISK_PENALTY_COEFF\b\s*=\s*)[0-9e.-]+", fr"\g<1>{params['RISK_PENALTY_COEFF']}", agent_code)
    agent_code = re.sub(r"(\blearning_rate\b\s*=\s*)[0-9e.-]+", fr"\g<1>{params['learning_rate']}", agent_code)
    agent_code = re.sub(r"(\bgamma\b\s*=\s*)[0-9e.-]+", fr"\g<1>{params['gamma']}", agent_code)
    agent_code = re.sub(r"(\bent_coef\b\s*=\s*)[0-9e.-]+", fr"\g<1>{params['ent_coef']}", agent_code)

    trial_id = uuid.uuid4()
    temp_agent_path = f"temp_agent_{trial_id}.py"; temp_results_path = f"temp_results_{trial_id}.json"
    with open(temp_agent_path, 'w') as f: f.write(agent_code)

    cagr = -1.0; status = "PENDING"; error_info = ""

    try:
        print(f"\n--- Starting Trial {trial.number} | Params: {params} ---")
        
        # --- MODIFICATION: Increased memory allocation from 4g to 8g ---
        docker_command = [
            'docker', 'run', '--rm', '--network=none', '--memory=8g', '--cpus=7.0',
            f'--volume={PROJECT_ROOT}:/app', f'--volume={os.path.join(PROJECT_ROOT, "data")}:/app/data:ro',
            DOCKER_IMAGE_NAME, 'python', temp_agent_path, '--output-path', temp_results_path
        ]
        # --- END MODIFICATION ---
        
        process = subprocess.Popen(docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        stderr_output = []
        print("--- Docker Output ---")
        for line in process.stdout: print(line, end='')
        process.wait()
        stderr_output = process.stderr.readlines()
        
        if process.returncode != 0:
            status = "FAIL"; error_info = "".join(stderr_output)
            print(f"--- ❌ Trial {trial.number} failed. Full stderr below: ---")
            print(error_info)
        else:
            status = "SUCCESS"
            with open(temp_results_path, 'r') as f:
                cagr = json.load(f).get('cagr', -1.0)
            print(f"--- ✅ Trial {trial.number} complete. CAGR: {cagr:.4f} ---")

    except Exception as e:
        status = "CRASH"; error_info = str(e)
        print(f"An unexpected error occurred in trial {trial.number}: {e}")
    finally:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\n--- Trial {trial.number} Finished ---\nTimestamp: {timestamp}\nStatus: {status}\nCAGR: {cagr:.4f}\nParameters:\n"
        for key, value in params.items(): log_entry += f"  - {key}: {value}\n"
        if error_info: log_entry += f"Error Info:\n{error_info}\n"
        log_entry += "-" * 25 + "\n"

        with open(TUNING_LOG_FILE, 'a') as log_file: log_file.write(log_entry)
        if os.path.exists(temp_agent_path): os.remove(temp_agent_path)
        if os.path.exists(temp_results_path): os.remove(temp_results_path)
            
    return cagr

if __name__ == "__main__":
    with open(TUNING_LOG_FILE, 'a') as log_file:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write("\n" + "="*50 + f"\n  NEW OPTUNA TUNING SESSION STARTED AT {start_time}\n" + "="*50 + "\n\n")
    print(f"Starting Optuna hyperparameter tuning for {N_TRIALS} trials.")
    print(f"Progress will be logged to '{TUNING_LOG_FILE}'.")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n\n" + "="*20 + " Tuning Complete " + "="*20)
    print(f"Best trial found: Trial #{study.best_trial.number}")
    print(f"  Value (Max CAGR): {study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items(): print(f"  {key}: {value}")
    print("\n" + "="*57 + "\n\nACTION: Please copy the 'Best Hyperparameters' and update the\n" + 
          "values at the top of your `agent_v0.py` file to make these\n" + 
          "the new default for your autonomous agent.\n" + "="*57)