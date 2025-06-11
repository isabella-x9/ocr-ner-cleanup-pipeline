import subprocess
import os

def run_ner_scorer(input_path):
    print(f"Running scorer on: {input_path}")
    result = subprocess.run(
        ["python", "ner_scorer.py", "--input", input_path],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)

