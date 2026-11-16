"""Score runs with LongMemEval's official evaluator (GPT-4o judge).

    pip install openai tqdm backoff numpy
    export OPENAI_API_KEY=...
    python score.py longmemeval_s_cleaned.json runs/random runs/vanilla

Per run this writes, next to hypotheses.jsonl:
    hypotheses.jsonl.eval-results-gpt-4o   per-question verdicts (autoeval_label.label)
    eval.log                               the evaluator's output, incl. accuracy per question type
"""
import json
import os
import subprocess
import sys

REPO = "LongMemEval"
JUDGE = "gpt-4o"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: python score.py DATA_PATH RUN_DIR [RUN_DIR ...]")
    data_path, run_dirs = os.path.abspath(sys.argv[1]), sys.argv[2:]
    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/xiaowu0162/{REPO}"], check=True)
    for run_dir in run_dirs:
        hyp_path = os.path.abspath(f"{run_dir}/hypotheses.jsonl")
        with open(f"{run_dir}/eval.log", "w") as log:
            subprocess.run(
                [sys.executable, "evaluate_qa.py", JUDGE, hyp_path, data_path],
                cwd=f"{REPO}/src/evaluation", stdout=log, check=True,
            )
        with open(f"{hyp_path}.eval-results-{JUDGE}") as f:
            labels = [json.loads(line)["autoeval_label"]["label"] for line in f]
        print(f"{run_dir}: accuracy {sum(labels) / len(labels):.3f} ({sum(labels)}/{len(labels)})")
