"""LongMemEval replay environment for context-editing controllers.

The recorded conversation is replayed turn by turn; after every exchange (a user turn plus the
recorded reply) the controller pulls one arm on the context. The frontier model is never called
while editing, so learning costs nothing beyond the arms themselves (edit) and whatever feedback
the controller buys in update(). It is called once per instance, in a separate phase, to answer
the question on the final edited context.

Setup:
    pip install -r requirements.txt
    export GEMINI_API_KEY=...          # only needed for `answer` and for the edit arm
    wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

Phase 1, replay (free for null/delete):
    python longmemeval_bandit.py replay longmemeval_s_cleaned.json vanilla --limit 60 --run-id exp01
    python longmemeval_bandit.py replay longmemeval_s_cleaned.json random  --limit 60 --run-id exp01
    # smaller haystacks: evidence sessions plus 8 sampled distractor sessions
    python longmemeval_bandit.py replay longmemeval_s_cleaned.json random --limit 60 --distractors 8

Phase 2, answer (one frontier call per instance):
    python longmemeval_bandit.py answer runs/random/exp01 --workers 8

Both phases resume: rerun the same command and finished instances are skipped.

Each run directory runs/<controller>/<run_id>/ holds:
    config.json        models, arms, seed and data settings
    steps.jsonl        one row per controller step (arm, target, token deltas, evidence hit)
    contexts.jsonl     final edited context and per-instance statistics
    transcript.txt     the controller's decisions, human-readable
    hypotheses.jsonl   final answers (written by `answer`)

Score with:
    python score.py longmemeval_s_cleaned.json runs/random/exp01 runs/vanilla/exp01
"""

import argparse
import json
import math
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import backoff

# (model id, thinking level)
CONTROLLED = ("gemini-3.8-flash", "medium")  # the frontier model whose context is being managed
EDITOR = ("gemini-3.8-flash", "low")         # rewrites a turn when the edit arm is pulled

EDIT_PROMPT = (
    "Rewrite this chat message as briefly as possible, keeping every fact, name, "
    "date and number. Output only the rewritten message.\n\n"
)
LME_FIELDS = {"question_id", "question", "question_date", "haystack_dates", "haystack_sessions"}

_client, _client_lock = None, threading.Lock()


def client():
    """Created on first use, so a null/delete replay needs neither the SDK nor an API key."""
    global _client
    with _client_lock:
        if _client is None:
            from google import genai
            _client = genai.Client()
    return _client


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def llm(contents, model):
    """contents: list of (role, [text, ...]). Returns (text, prompt token count).

    An empty reply raises, so it is retried and never recorded as an answer.
    """
    from google.genai import types
    model_id, thinking_level = model
    response = client().models.generate_content(
        model=model_id,
        contents=[
            types.Content(role=role, parts=[types.Part(text=text) for text in texts])
            for role, texts in contents
        ],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
        ),
    )
    if not response.text:
        raise RuntimeError(f"{model_id} returned no text")
    return response.text, getattr(response.usage_metadata, "prompt_token_count", None)


# --- context -----------------------------------------------------------------------------
# A turn is {"role", "text", "date", "session", "evidence", "edited"}. The session date is
# metadata, not text, so deleting a session's first turn cannot erase the date of the rest.

def render(turns, question=None):
    """Context -> API contents. Adds a date header per surviving session and merges
    consecutive same-role turns, which deletions create."""
    contents, session = [], None

    def add(role, text):
        if contents and contents[-1][0] == role:
            contents[-1][1].append(text)
        else:
            contents.append((role, [text]))

    for turn in turns:
        if turn["session"] != session:
            session = turn["session"]
            add("user", f"[New session, {turn['date']}]")
        add(turn["role"], turn["text"])
    if question:
        add("user", question)
    return contents


def tokens(turns, question=None):
    """Estimated input tokens (chars / 4). `answer` records the real count of the final call."""
    return math.ceil(sum(len(text) for _, texts in render(turns, question) for text in texts) / 4)


def final_question(instance):
    return f"[Current date: {instance['question_date']}]\n{instance['question']}"


def exchanges(instance, distractors, rng):
    """The recorded haystack as a list of exchanges; each is a user turn plus the replies to it.

    With distractors=K, only the evidence sessions and K sampled other sessions are kept.
    """
    sessions, dates = instance["haystack_sessions"], instance["haystack_dates"]
    keep = range(len(sessions))
    if distractors is not None:
        ids = instance.get("haystack_session_ids") or list(keep)
        evidence_ids = set(instance.get("answer_session_ids") or [])
        other = [k for k in keep if ids[k] not in evidence_ids]
        sampled = set(rng.sample(other, min(distractors, len(other))))
        keep = [k for k in keep if ids[k] in evidence_ids or k in sampled]
    out = []
    for k in keep:
        for n, turn in enumerate(sessions[k]):
            role = "user" if turn["role"] == "user" else "model"
            if role == "user" or n == 0:
                out.append([])
            out[-1].append({
                "role": role, "text": turn["content"], "date": dates[k], "session": k,
                "evidence": bool(turn.get("has_answer")), "edited": False,
            })
    return out


# --- arms: each takes the context and the index of the selected turn ----------------------
def null(turns, i):
    pass


def delete(turns, i):
    del turns[i]


def edit(turns, i):
    text, _ = llm([("user", [EDIT_PROMPT + turns[i]["text"]])], EDITOR)
    turns[i]["text"], turns[i]["edited"] = text, True


ARMS = {"null": null, "delete": delete, "edit": edit}


# --- controllers --------------------------------------------------------------------------
class Controller:
    """One instance lives for the whole run, so a learned policy keeps its state across
    conversations. Subclass this for the GP bandit."""

    def __init__(self, arms):
        self.arms = arms

    def act(self, turns, t, T, rng):
        """Called after exchange t of T - 1 (the T-th frontier call is the final question).
        Return (arm name, turn index); the index is None for null."""
        return "null", None

    def update(self, row):
        """Called after the arm is applied, with the steps.jsonl row: delta_tokens,
        remaining_calls, token_saving = delta_tokens * (T - t), target text before/after.
        Buy feedback here (e.g. one judge call per non-null action) and learn from it."""


class Vanilla(Controller):
    pass


class Random(Controller):
    def act(self, turns, t, T, rng):
        arm = rng.choice(self.arms)
        return (arm, None) if arm == "null" else (arm, rng.randrange(len(turns)))


CONTROLLERS = {"vanilla": Vanilla, "random": Random}


# --- phase 1: replay ----------------------------------------------------------------------
def replay(instance, controller, args, log_step, log):
    qid = instance["question_id"]
    rng = random.Random(f"{args.seed}:{qid}")  # per instance: independent of order and --limit
    conversation = exchanges(instance, args.distractors, rng)
    question = final_question(instance)
    T = len(conversation) + 1  # frontier calls a live chat would make: one per exchange + the question
    turns, unedited = [], []
    k = k_unedited = 0         # total input tokens over those T calls, with and without editing
    log(f"\n{'=' * 80}\n{qid}  ({len(conversation)} exchanges)\n{'=' * 80}\n")
    for t, exchange in enumerate(conversation, 1):
        users = [turn for turn in exchange if turn["role"] == "user"]
        k += tokens(turns + users)          # what the frontier call at exchange t would have read
        k_unedited += tokens(unedited + users)
        turns.extend(dict(turn) for turn in exchange)
        unedited.extend(exchange)
        before = tokens(turns)
        arm, i = controller.act(turns, t, T, rng)
        target = None if i is None else dict(turns[i])
        ARMS[arm](turns, i)
        delta = before - tokens(turns)
        row = {
            "question_id": qid, "t": t, "T": T, "arm": arm, "turn": i,
            "role": target and target["role"], "evidence": bool(target and target["evidence"]),
            "delta_tokens": delta, "remaining_calls": T - t, "token_saving": delta * (T - t),
            "context_turns": len(turns), "context_tokens": before - delta,
            "text_before": target and target["text"],
            "text_after": turns[i]["text"] if arm == "edit" else None,
        }
        controller.update(row)
        log_step(row)
        if target:
            log(
                f"t={t}/{T - 1} {arm} turn {i} ({target['role']}"
                f"{', EVIDENCE' if target['evidence'] else ''}) {delta:+d} tok saved\n"
                f"    before: {target['text'][:200]!r}\n"
                + (f"    after:  {turns[i]['text'][:200]!r}\n" if arm == "edit" else "")
            )
    k += tokens(turns, question)
    k_unedited += tokens(unedited, question)
    evidence = [turn for turn in turns if turn["evidence"]]
    return {
        "question_id": qid, "question_type": instance.get("question_type"), "question": question,
        "exchanges": len(conversation),
        "context_tokens": tokens(turns, question), "unedited_tokens": tokens(unedited, question),
        "k_tokens": k, "k_unedited_tokens": k_unedited,
        "evidence_total": sum(turn["evidence"] for turn in unedited),
        "evidence_kept": len(evidence), "evidence_edited": sum(turn["edited"] for turn in evidence),
        "turns": turns,
    }


def select(data, limit, stratify, seed):
    """First `limit` instances, or a seeded round-robin over question types."""
    if limit is None or not stratify:
        return data[:limit]
    rng, groups = random.Random(seed), {}
    for instance in data:
        groups.setdefault(instance.get("question_type"), []).append(instance)
    for group in groups.values():
        rng.shuffle(group)
    picked = []
    while len(picked) < limit and any(groups.values()):
        for key in sorted(groups, key=str):
            if groups[key] and len(picked) < limit:
                picked.append(groups[key].pop())
    return picked


def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["question_id"] for line in f if line.strip()}


def write_row(file, row):
    file.write(json.dumps(row) + "\n")
    file.flush()


def run_replay(args):
    with open(args.data) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise SystemExit("DATA must contain a non-empty LongMemEval JSON array")
    missing = LME_FIELDS - data[0].keys()
    if missing:
        raise SystemExit(f"not a LongMemEval file, missing fields: {missing}")
    arms = args.arms.split(",")
    if set(arms) - ARMS.keys():
        raise SystemExit(f"unknown arms {set(arms) - ARMS.keys()}; choose from {', '.join(ARMS)}")
    data = select(data, args.limit, not args.no_stratify, args.seed)
    out_dir = os.path.join("runs", args.controller, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    config = {
        "controller": args.controller, "arms": arms, "seed": args.seed, "data": args.data,
        "limit": args.limit, "stratify": not args.no_stratify, "distractors": args.distractors,
        "controlled": CONTROLLED, "editor": EDITOR,
    }
    config_path = f"{out_dir}/config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            if json.load(f) != json.loads(json.dumps(config)):
                raise SystemExit(f"{out_dir} holds a run with different settings; pick another --run-id")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    done = done_ids(f"{out_dir}/contexts.jsonl")
    if done:
        # A learned controller restarts cold here; its state is not checkpointed.
        print(f"Resuming: {len(done)} instances already replayed", file=sys.stderr)
    print(f"Writing run outputs to {out_dir}", file=sys.stderr)
    controller = CONTROLLERS[args.controller](arms)
    with (
        open(f"{out_dir}/steps.jsonl", "a") as steps,
        open(f"{out_dir}/contexts.jsonl", "a") as contexts,
        open(f"{out_dir}/transcript.txt", "a") as transcript,
    ):
        def log(text):
            transcript.write(text)
            if args.verbose:
                print(text, end="", file=sys.stderr, flush=True)

        for instance in data:
            if instance["question_id"] in done:
                continue
            # Buffer the steps so a crash mid-instance leaves no partial rows behind.
            rows = []
            result = replay(instance, controller, args, rows.append, log)
            for row in rows:
                write_row(steps, row)
            write_row(contexts, result)
            transcript.flush()
            print(
                f"{args.controller}: {result['question_id']} {result['context_tokens']}"
                f"/{result['unedited_tokens']} tok, evidence {result['evidence_kept']}"
                f"/{result['evidence_total']}",
                file=sys.stderr,
            )


# --- phase 2: answer ----------------------------------------------------------------------
def run_answer(args):
    contexts_path, hyp_path = f"{args.run_dir}/contexts.jsonl", f"{args.run_dir}/hypotheses.jsonl"
    if not os.path.exists(contexts_path):
        raise SystemExit(f"{contexts_path} not found; run `replay` first")
    done = done_ids(hyp_path)
    with open(contexts_path) as f:
        todo = [row for row in map(json.loads, f) if row["question_id"] not in done]
    print(f"{len(done)} answered, {len(todo)} to go", file=sys.stderr)
    lock, failed = threading.Lock(), []

    with open(hyp_path, "a") as hyps:
        def answer(row):
            try:
                text, prompt_tokens = llm(render(row["turns"], row["question"]), CONTROLLED)
            except Exception as error:  # left unanswered, so rerunning retries it
                failed.append(row["question_id"])
                print(f"FAILED {row['question_id']}: {error}", file=sys.stderr)
                return
            with lock:
                write_row(hyps, {
                    "question_id": row["question_id"], "hypothesis": text,
                    "prompt_tokens": prompt_tokens,
                })
            print(f"{row['question_id']} done ({prompt_tokens} prompt tokens)", file=sys.stderr)

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(answer, todo))
    if failed:
        raise SystemExit(f"{len(failed)} instances failed; rerun the same command to retry them")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    phases = parser.add_subparsers(dest="phase", required=True)
    p = phases.add_parser("replay", help="replay conversations and let the controller edit")
    p.add_argument("data")
    p.add_argument("controller", choices=CONTROLLERS)
    p.add_argument("--limit", type=int, help="number of instances")
    p.add_argument("--no-stratify", action="store_true", help="take the first LIMIT instances")
    p.add_argument("--distractors", type=int, help="keep evidence sessions plus this many others")
    p.add_argument("--arms", default="null,delete", help="comma-separated; edit costs one call per pull")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--verbose", action="store_true", help="echo decisions to stderr")
    p = phases.add_parser("answer", help="answer each question on its final context")
    p.add_argument("run_dir")
    p.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "replay":
        if args.limit is not None and args.limit <= 0:
            raise SystemExit("--limit must be positive")
        if not args.run_id or args.run_id in {".", ".."} or "/" in args.run_id or os.sep in args.run_id:
            raise SystemExit("--run-id must be a directory name without path separators")
        run_replay(args)
    else:
        run_answer(args)