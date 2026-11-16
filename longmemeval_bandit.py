"""LongMemEval chat with a 3-arm context bandit (null / delete / edit) after each LLM turn.

Setup:
    pip install -r requirements.txt
    export GEMINI_API_KEY=...
    wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

Run (controller is "random" or "vanilla"; the optional arguments limit instances and name the run):
    python longmemeval_bandit.py longmemeval_s_cleaned.json random 20
    python longmemeval_bandit.py longmemeval_s_cleaned.json vanilla 20
    python longmemeval_bandit.py longmemeval_s_cleaned.json random 20 experiment-01

Each run writes to runs/<controller>/<run_id>/, where run_id defaults to a timestamp:
    hypotheses.jsonl   the controlled model's final answer per question
    steps.jsonl        one row per bandit step (arm, target turn, context size after the action)
    transcript.txt     human-readable chats and bandit decisions (also streamed to stderr)

Score with:
    python score.py longmemeval_s_cleaned.json runs/random/<run_id> runs/vanilla/<run_id>
"""

import json
import os
import random
import sys
from datetime import datetime

from google import genai
from google.genai import types

# (model id, thinking level)
CONTROLLED = ("gemini-3.8-flash", "medium")  # the frontier model whose context is being managed
EDITOR = ("gemini-3.8-flash", "low")         # rewrites a turn when the edit arm is pulled

EDIT_PROMPT = (
    "Rewrite this chat message as briefly as possible, keeping every fact, name, "
    "date and number. Output only the rewritten message.\n\n"
)
LME_FIELDS = {"question_id", "question", "question_date", "haystack_dates", "haystack_sessions"}
client = genai.Client()


def llm(turns, model):
    """turns: list of {"role": "user" | "model", "text": str}."""
    model_id, thinking_level = model
    response = client.models.generate_content(
        model=model_id,
        contents=[types.Content(role=t["role"], parts=[types.Part(text=t["text"])]) for t in turns],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
        ),
    )
    return response.text or ""


# --- arms: each takes the context and the index of the selected turn T ---
def null(turns, i):
    pass


def delete(turns, i):
    del turns[i]


def edit(turns, i):
    prompt = [{"role": "user", "text": EDIT_PROMPT + turns[i]["text"]}]
    turns[i]["text"] = llm(prompt, EDITOR) or turns[i]["text"]


ARMS = [null, delete, edit]


# --- controllers: context -> (arm, turn index). Add the learned policy here. ---
def random_controller(turns):
    return random.choice(ARMS), random.randrange(len(turns))


def vanilla_controller(turns):
    return null, None


CONTROLLERS = {"random": random_controller, "vanilla": vanilla_controller}


def run(instance, controller, log_step, log_transcript):
    turns, step = [], 0
    log_transcript(
        f"\n{'=' * 80}\n"
        f"QUESTION ID: {instance['question_id']}\n"
        f"{'=' * 80}\n"
    )
    for date, session in zip(instance["haystack_dates"], instance["haystack_sessions"]):
        for n, user in enumerate(t for t in session if t["role"] == "user"):
            text = f"[New session, {date}]\n{user['content']}" if n == 0 else user["content"]
            turns.append({"role": "user", "text": text})
            log_transcript(f"\n--- USER TURN (session date: {date}) ---\n{text}\n")
            model_text = llm(turns, CONTROLLED)
            turns.append({"role": "model", "text": model_text})
            log_transcript(f"\n--- MODEL TURN ---\n{model_text}\n")
            arm, i = controller(turns)
            role = None if i is None else turns[i]["role"]
            selected_text = None if i is None else turns[i]["text"]
            arm(turns, i)
            decision = (
                f"arm={arm.__name__}, target_turn={i}, target_role={role}"
                if i is not None
                else f"arm={arm.__name__}, target_turn=None"
            )
            log_transcript(
                f"\n--- BANDIT DECISION ---\n{decision}\n"
                + (f"TARGET CONTENT:\n{selected_text}\n" if selected_text is not None else "")
            )
            if arm is edit and i is not None:
                log_transcript(
                    f"\n--- EDITED TURN ---\n"
                    f"BEFORE:\n{selected_text}\n\nAFTER:\n{turns[i]['text']}\n"
                )
            log_step({
                "question_id": instance["question_id"], "step": step,
                "arm": arm.__name__, "turn": i, "role": role,  # turn = index when the arm was pulled
                "context_turns": len(turns), "context_chars": sum(len(t["text"]) for t in turns),
            })
            step += 1
    question = f"[Current date: {instance['question_date']}]\n{instance['question']}"
    log_transcript(f"\n--- QUESTION ---\n{question}\n")
    answer = llm(turns + [{"role": "user", "text": question}], CONTROLLED)
    log_transcript(f"\n--- FINAL ANSWER ---\n{answer}\n")
    return answer


def check_setup(data):
    """Fail fast: the file looks like LongMemEval and both model configs answer."""
    missing = LME_FIELDS - data[0].keys()
    assert not missing, f"not a LongMemEval file, missing fields: {missing}"
    for name, model in [("controlled", CONTROLLED), ("editor", EDITOR)]:
        assert llm([{"role": "user", "text": "Reply with OK."}], model), f"{name} {model} returned nothing"
        print(f"{name}: {model[0]} @ thinking_level={model[1]} OK", file=sys.stderr)
    print(f"LongMemEval: {len(data)} instances OK", file=sys.stderr)


def write_row(file, row):
    file.write(json.dumps(row) + "\n")
    file.flush()


def write_transcript(file, text):
    """Write transcript text to disk and stderr so progress is visible live."""
    file.write(text)
    file.flush()
    print(text, end="", file=sys.stderr, flush=True)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4, 5):
        raise SystemExit(
            "usage: python longmemeval_bandit.py DATA_PATH {random,vanilla} [LIMIT] [RUN_ID]"
        )
    data_path, name = sys.argv[1:3]
    if name not in CONTROLLERS:
        raise SystemExit(f"unknown controller {name!r}; choose from {', '.join(CONTROLLERS)}")
    limit = int(sys.argv[3]) if len(sys.argv) == 4 else None
    if len(sys.argv) == 5:
        limit = int(sys.argv[3])
        run_id = sys.argv[4]
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if limit is not None and limit <= 0:
        raise SystemExit("LIMIT must be a positive integer")
    if not run_id or run_id in {".", ".."} or "/" in run_id or os.sep in run_id:
        raise SystemExit("RUN_ID must be a non-empty directory name without path separators")
    with open(data_path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise SystemExit("DATA_PATH must contain a non-empty LongMemEval JSON array")
    data = data[:limit]
    check_setup(data)
    random.seed(0)
    out_dir = os.path.join("runs", name, run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing run outputs to {out_dir}", file=sys.stderr)
    with (
        open(f"{out_dir}/hypotheses.jsonl", "w") as hyps,
        open(f"{out_dir}/steps.jsonl", "w") as steps,
        open(f"{out_dir}/transcript.txt", "w") as transcript,
    ):
        for instance in data:
            answer = run(
                instance,
                CONTROLLERS[name],
                lambda row: write_row(steps, row),
                lambda text: write_transcript(transcript, text),
            )
            write_row(hyps, {"question_id": instance["question_id"], "hypothesis": answer})
            print(f"{name}: {instance['question_id']} done", file=sys.stderr)
