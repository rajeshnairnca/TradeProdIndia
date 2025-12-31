import os
import google.genai as genai
from dotenv import load_dotenv
import difflib

load_dotenv()

client = genai.Client()

def clean_llm_code(llm_response: str) -> str:
    """Cleans the raw response from the LLM, removing markdown code blocks."""
    if not llm_response:
        return ""
    cleaned_code = llm_response.strip()
    if cleaned_code.startswith("```python"):
        cleaned_code = cleaned_code[len("```python"):].strip()
    if cleaned_code.endswith("```"):
        cleaned_code = cleaned_code[:-len("```")].strip()
    return cleaned_code

def get_new_alpha_idea(existing_alphas_summary: str, ensemble_cagr: float, failed_attempts_summary: str, data_schema: str, guidance: str = "") -> tuple[str, str]:
    """Asks the Gemini model to generate a new, diverse rule-based strategy. Optional guidance can steer the idea."""
    guidance_text = f"\n\nADDITIONAL GUIDANCE FROM USER:\n{guidance}\n" if guidance else ""
    prompt = f"""
    You are an expert quantitative researcher tasked with designing a portfolio of diverse alpha strategies.

    Here is a summary of the alphas already in our portfolio:
    {existing_alphas_summary}

    Their combined performance gives a CAGR of {ensemble_cagr:.2f}%.

    To avoid repeating past mistakes, here is a summary of previously attempted strategies that FAILED to improve the ensemble and were discarded. Do not suggest these ideas again:
    {failed_attempts_summary}

    {guidance_text}

    Your task is to propose a NEW, CREATIVE, fully deterministic rule-based alpha strategy that is likely to be UNCORRELATED with the existing successful ones and different from the failed attempts.

    You must provide your answer as a Python script containing three things:
    1. A function `generate_scores(df: pd.DataFrame) -> pd.Series` that takes the full market DataFrame and returns a numeric score for every row (same index).
    2. A string variable `DESCRIPTION` containing a one-sentence summary of the new alpha's strategy.
    3. A list variable `REGIME_TAGS` describing which regimes this strategy is intended for.

    **Data Schema Reference:**
    The pandas DataFrame `full_data` that you will be manipulating has the following columns. Use these exact names in your code:
    `{data_schema}`
    The DataFrame index is a MultiIndex with levels: date, ticker.

    CRITICAL INSTRUCTIONS:
    - Your response MUST be ONLY a Python script with the `generate_scores` function, `DESCRIPTION`, and `REGIME_TAGS`.
    - DO NOT include any other code, explanations, or markdown.
    - The `generate_scores` function must be deterministic (no randomness) and return a `pd.Series` indexed exactly like `df`.
    - Avoid look-ahead bias: do not use negative shifts, forward-looking rolling windows, or future data.
    - `REGIME_TAGS` must be a list containing any of: bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol.
    """
    response = client.models.generate_content(
        # model="gemini-2.5-pro",
        model="gemini-3-pro-preview",
        contents=prompt
    )
    suggested_code = response.text
    return prompt, suggested_code

def summarize_code_change(old_code, new_code):
    """Asks the LLM to summarize the difference between two pieces of code."""
    diff = difflib.unified_diff(
        old_code.splitlines(keepends=True),
        new_code.splitlines(keepends=True),
        fromfile='old_agent.py',
        tofile='new_agent.py',
    )
    diff_text = "".join(diff)

    if not diff_text:
        return "No code changes were made.", ""

    prompt = f"""
    You are a code analysis expert. Below is a 'diff' output showing the changes between two Python scripts.
    Please summarize the main semantic change in a single, concise sentence.
    Focus on what the change does, not just the line numbers.
    Here is the diff:
    {diff_text}
    ```"""
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )
    summary = response.text.strip()
    return summary, diff_text

def get_code_fix(broken_code, error_traceback):
    """Asks the Gemini model to provide a debugged version of code."""
    prompt = f"""
    You are an expert Python debugger. The following Python script failed to execute.
    Analyze the error traceback and the code, identify the bug, and provide a fixed version of the full script.

    THE FAILED CODE:```python
    {broken_code}
    ```

    THE ERROR MESSAGE:
    ```
    {error_traceback}
    ```

    CRITICAL INSTRUCTIONS:
    1. Your response MUST be ONLY the complete, raw Python code for the fixed script.
    2. DO NOT include any explanations or markdown.
    """
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )
    fixed_code = response.text
    return fixed_code
