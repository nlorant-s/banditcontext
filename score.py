"""Score runs with LongMemEval's judge prompts and a judge model of your choice.

    export GEMINI_API_KEY=...
    python score.py longmemeval_s_cleaned.json runs/random/exp01 runs/vanilla/exp01

The prompts are LongMemEval's own, word for word; only the judge model differs from the paper
(gpt-4o-2024-08-06, which OpenAI retires on 2027-01-14). To report how closely your judge tracks
the paper's while that model still exists, score the same answers with both:

    export OPENAI_API_KEY=...            # pip install openai
    python score.py DATA RUN_DIR --compare openai:gpt-4o-2024-08-06

A judge is "<gemini model>:<thinking level>" or "openai:<model>". Per run and judge this writes
judgments.<judge>.jsonl (question_id, label, raw judge reply) and resumes if rerun; delete the
file to re-judge. It writes the matching summary to results.<judge>.txt, and also prints accuracy,
overall and per question type, beside compression and evidence retention.

One deviation from the official script: it labels a reply correct if it contains "yes" anywhere,
which is safe only under its 10-token cap. Here a reply must start with "yes"; replies starting
with neither yes nor no are counted as wrong and reported as unclear.
"""
import argparse
import json
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from longmemeval_bandit import llm

JUDGE = "gemini-3.8-flash:low"

# Judge prompts from LongMemEval, src/evaluation/evaluate_qa.py (commit 9e0b455), reproduced
# verbatim. MIT License, Copyright (c) 2024 Di Wu. https://github.com/xiaowu0162/LongMemEval
_HEAD = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. "
_STEPS = "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. "
_TAIL = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_FACT = _HEAD + _STEPS + _TAIL
TEMPLATES = {
    "single-session-user": _FACT,
    "single-session-assistant": _FACT,
    "multi-session": _FACT,
    "temporal-reasoning": _HEAD + _STEPS + "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. " + _TAIL,
    "knowledge-update": _HEAD + "If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer." + _TAIL,
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
}
ABSTENTION = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."


def judge_prompt(reference, hypothesis):
    """Abstention questions are the ones whose id contains "_abs", as in the official script."""
    abstention = "_abs" in reference["question_id"]
    template = ABSTENTION if abstention else TEMPLATES[reference["question_type"]]
    return template.format(reference["question"], reference["answer"], hypothesis)


def ask(judge, prompt):
    provider, _, rest = judge.partition(":")
    if provider == "openai":  # the official call: temperature 0, 10 output tokens
        from openai import OpenAI
        completion = OpenAI(max_retries=5).chat.completions.create(
            model=rest, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=10,
        )
        return completion.choices[0].message.content or ""
    if not rest:
        raise SystemExit(f"judge {judge!r}: expected '<gemini model>:<thinking level>' or 'openai:<model>'")
    return llm([("user", [prompt])], (provider, rest))[0]


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_run(run_dir, references, judge, workers):
    """Judge every answer in the run that this judge has not already judged.
    Returns question_id -> judgment row."""
    path = f"{run_dir}/judgments.{re.sub(r'[^A-Za-z0-9.-]+', '_', judge)}.jsonl"
    judged = {row["question_id"]: row for row in read_jsonl(path)}
    hypotheses = read_jsonl(f"{run_dir}/hypotheses.jsonl")
    if not hypotheses:
        raise SystemExit(f"{run_dir}/hypotheses.jsonl is missing or empty; run `answer` first")
    unknown = [h["question_id"] for h in hypotheses if h["question_id"] not in references]
    if unknown:
        print(f"  WARNING: {len(unknown)} answers are not in the reference data and are skipped")
    todo = [h for h in hypotheses if h["question_id"] in references and h["question_id"] not in judged]
    lock = threading.Lock()

    with open(path, "a") as out:
        def judge_one(hypothesis):
            qid = hypothesis["question_id"]
            reply = ask(judge, judge_prompt(references[qid], hypothesis["hypothesis"])).strip()
            verdict = reply.lower()
            row = {
                "question_id": qid, "label": verdict.startswith("yes"),
                "unclear": not verdict.startswith(("yes", "no")), "reply": reply, "judge": judge,
            }
            with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
                judged[qid] = row

        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(judge_one, todo))
    return {h["question_id"]: judged[h["question_id"]] for h in hypotheses if h["question_id"] in judged}


def summarise(run_dir, judge, judged, references):
    stats = {row["question_id"]: row for row in read_jsonl(f"{run_dir}/contexts.jsonl")}
    correct = sum(row["label"] for row in judged.values())
    lines = [f"{run_dir}  [{judge}]: accuracy {correct / len(judged):.3f} ({correct}/{len(judged)})"]
    unclear = sum(row["unclear"] for row in judged.values())
    if unclear:
        lines.append(f"  WARNING: {unclear} judge replies were neither yes nor no (counted as wrong)")
    if len(judged) < len(stats):
        lines.append(f"  WARNING: {len(stats) - len(judged)} replayed instances have no scored answer")
    scored = [stats[qid] for qid in judged if qid in stats]
    if scored:
        total = lambda key: sum(row[key] for row in scored)
        lines.append(f"  final context   {total('context_tokens') / total('unedited_tokens'):.1%} of unedited (est. tokens)")
        lines.append(f"  total input K   {total('k_tokens') / total('k_unedited_tokens'):.1%} of unedited (est. tokens)")
        if total("evidence_total"):
            lines.append(f"  evidence kept   {total('evidence_kept') / total('evidence_total'):.1%} of evidence turns"
                         f" ({total('evidence_edited')} of them edited)")
    by_type = defaultdict(list)
    for qid, row in judged.items():
        question_type = references[qid]["question_type"]
        by_type["abstention (_abs)" if "_abs" in qid else question_type].append(row["label"])
    for question_type, labels in sorted(by_type.items()):
        lines.append(f"  {question_type:<28} {sum(labels) / len(labels):.3f} ({sum(labels)}/{len(labels)})")

    summary = "\n".join(lines)
    print("\n" + summary)
    result_path = f"{run_dir}/results.{re.sub(r'[^A-Za-z0-9.-]+', '_', judge)}.txt"
    with open(result_path, "w") as out:
        out.write(summary + "\n")


def compare(judge, judged, other_judge, other_judged):
    """Agreement between two judges on the answers both have judged."""
    shared = [qid for qid in judged if qid in other_judged]
    pairs = [(judged[qid]["label"], other_judged[qid]["label"]) for qid in shared]
    agree = sum(a == b for a, b in pairs)
    print(f"  agreement with {other_judge}: {agree / len(pairs):.1%} ({agree}/{len(pairs)})")
    print(f"    {judge} yes, {other_judge} no: {sum(a and not b for a, b in pairs)}")
    print(f"    {judge} no, {other_judge} yes: {sum(b and not a for a, b in pairs)}")
    return [qid for qid, (a, b) in zip(shared, pairs) if a != b]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data")
    parser.add_argument("run_dirs", nargs="+", metavar="run_dir")
    parser.add_argument("--judge", default=JUDGE)
    parser.add_argument("--compare", metavar="JUDGE", help="also judge with this model and report agreement")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with open(args.data) as f:
        references = {entry["question_id"]: entry for entry in json.load(f)}
    for run_dir in args.run_dirs:
        run_dir = run_dir.rstrip("/")
        judged = judge_run(run_dir, references, args.judge, args.workers)
        summarise(run_dir, args.judge, judged, references)
        if args.compare:
            other = judge_run(run_dir, references, args.compare, args.workers)
            disagreements = compare(args.judge, judged, args.compare, other)
            if disagreements:
                print(f"    disagreed on: {' '.join(disagreements[:20])}{' ...' if len(disagreements) > 20 else ''}")