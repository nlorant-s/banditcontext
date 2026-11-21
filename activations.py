"""Turn text -> residual-stream activation vectors from a small local Hugging Face model, as
context features for a GP bandit controller in longmemeval_bandit.py.

Each chat turn is run once through a small decoder LM; the hidden state after transformer block L
is mean-pooled over the turn's tokens (the first token, an attention sink with outsized norm, is
left out). Each instance's final question is encoded the same way, once. Raw vectors (1024-d for
Qwen3-0.6B) are too wide for a GP kernel, so a Projector, fitted once on held-out conversations,
standardises them and keeps the top `dim` whitened PCA components. At each step the GP input is
one row per candidate turn, built from the turn vector and the question vector by one of
REPRESENTATIONS (default "interaction"; trade-offs documented there), followed by a few cheap
scalar features. The question is oracle knowledge, see QUESTION ORACLE in longmemeval_bandit.py;
with the oracle off the featurizer falls back to the question-blind "turn" representation.

Setup (beside the repo's own requirements):
    pip install torch transformers scikit-learn

1. Feasibility check and layer choice. Probes several layers for whether a turn's activation
   predicts that it is evidence for its instance's final question (turn alone, [turn | question],
   and their cosine), on instances held out from the ones `replay --limit 60 --seed 0` will use:
    python activations.py probe longmemeval_s_cleaned.json --instances 30 --distractors 8 \\
        --holdout-limit 60 --layers 4,8,12,14,16,18,20,24,28

2. Fit the projector at the chosen layer (same holdout rule):
    python activations.py fit longmemeval_s_cleaned.json --layer 16 --dim 32 \\
        --instances 30 --holdout-limit 60 --out runs/proj_qwen3-0.6b_L16_d32.npz

3. Use it in a controller:
    from activations import load_featurizer
    class GPBandit(Controller):
        def __init__(self, arms, oracle=True):  # oracle must match replay's --question-oracle
            super().__init__(arms)
            self.featurize = load_featurizer("runs/proj_qwen3-0.6b_L16_d32.npz", oracle=oracle)
        def begin(self, question_id, question): # question is None with --no-question-oracle
            self.featurize.begin(question)
        def act(self, turns, t, T, rng):
            X = self.featurize(turns, t, T)     # (len(turns), 2 * 32 + 2 + 5) for "interaction"
            ...                                 # GP-UCB over (row of X, arm); null if no gain

Ad hoc:
    python activations.py encode "I adopted a beagle named Rufus last May." --layer 16
"""

import argparse
import hashlib
import json
import math
import random
import sys

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"   # 28 blocks, hidden size 1024, Apache-2.0
META_FEATURES = ["is_user", "log_tokens", "age", "edited", "remaining"]


def default_layer(num_layers):
    """About 60% of the depth: block 17 of 28 for Qwen3-0.6B. The probe decides for real."""
    return max(1, round(0.6 * num_layers))


# --- encoder ------------------------------------------------------------------------------
def _find_blocks(model):
    """The ModuleList of transformer blocks: (parent module, attribute name, list)."""
    import torch.nn as nn
    config = getattr(model.config, "text_config", model.config)
    n = config.num_hidden_layers
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == n:
            parent_name, _, attr = name.rpartition(".")
            return model.get_submodule(parent_name) if parent_name else model, attr, module
    raise RuntimeError(f"no ModuleList of {n} blocks found in {type(model).__name__}")


class TurnEncoder:
    """Text -> pooled hidden state after block `layer` (1-based; -1 is the last block).

    Blocks above the deepest requested layer are dropped, so they cost nothing. Results are cached
    by (layer, text), so re-scoring the whole context every step only encodes new or edited turns.
    """

    def __init__(self, model=DEFAULT_MODEL, layers=None, pooling="mean", max_tokens=512,
                 batch_size=8, device=None, dtype=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model_name, self.pooling, self.max_tokens = model, pooling, max_tokens
        self.batch_size, self.device = batch_size, device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            net = AutoModel.from_pretrained(model, dtype=dtype)
        except TypeError:  # transformers < 4.56
            net = AutoModel.from_pretrained(model, torch_dtype=dtype)
        parent, attr, blocks = _find_blocks(net)
        self.num_layers = len(blocks)
        if layers is None:
            layers = [default_layer(self.num_layers)]
        self.layers = [l if l > 0 else self.num_layers + 1 + l for l in layers]
        bad = [l for l in self.layers if not 1 <= l <= self.num_layers]
        if bad:
            raise ValueError(f"layers {bad} out of range 1..{self.num_layers} for {model}")
        setattr(parent, attr, torch.nn.ModuleList(list(blocks)[:max(self.layers)]))
        self._captured = {}
        for layer in self.layers:
            def hook(module, inputs, output, layer=layer):
                self._captured[layer] = output[0] if isinstance(output, (tuple, list)) else output
            getattr(parent, attr)[layer - 1].register_forward_hook(hook)
        self.model = net.to(device).eval()
        self.dim = net.config.get_text_config().hidden_size
        self.cache = {}

    @staticmethod
    def _key(layer, text):
        return layer, hashlib.sha1(text.encode()).hexdigest()

    def encode_layers(self, texts, progress=False):
        """{layer: float32 array (len(texts), hidden size)}."""
        texts = [text if text.strip() else " " for text in texts]
        todo = sorted({text for text in texts
                       if any(self._key(l, text) not in self.cache for l in self.layers)},
                      key=len)  # similar lengths per batch, less padding
        batches = range(0, len(todo), self.batch_size)
        if progress:
            from tqdm import tqdm
            batches = tqdm(batches, desc="encoding", unit="batch")
        for start in batches:
            batch = todo[start:start + self.batch_size]
            for layer, vectors in self._forward(batch).items():
                for text, vector in zip(batch, vectors):
                    self.cache[self._key(layer, text)] = vector
        return {l: np.stack([self.cache[self._key(l, text)] for text in texts]) if texts
                else np.zeros((0, self.dim), np.float32) for l in self.layers}

    def encode(self, texts, progress=False):
        """Float32 array (len(texts), hidden size) at the first requested layer."""
        return self.encode_layers(texts, progress)[self.layers[0]]

    def _forward(self, batch):
        torch = self.torch
        enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                             max_length=self.max_tokens).to(self.device)
        self._captured.clear()
        with torch.inference_mode():
            self.model(**enc, use_cache=False)
        mask = enc["attention_mask"].float()
        lengths = mask.sum(1).long().clamp(min=1)  # a text with no tokens pools to ~zeros
        out = {}
        for layer in self.layers:
            hidden = self._captured[layer].float()
            if self.pooling == "last":
                pooled = hidden[torch.arange(len(batch), device=hidden.device), lengths - 1]
            else:
                m = mask.clone()
                m[lengths > 1, 0] = 0  # drop the attention-sink first token
                pooled = (hidden * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            out[layer] = pooled.cpu().numpy().astype(np.float32)
        return out


# --- projection to GP-sized features -------------------------------------------------------
class Projector:
    """Per-dimension standardisation (tames the few massive-activation dimensions), then the top
    `dim` PCA components scaled to unit variance, so one GP lengthscale prior fits them all."""

    def __init__(self, mean, scale, components, std, meta=None):
        self.mean, self.scale, self.components, self.std = mean, scale, components, std
        self.meta = meta or {}

    @classmethod
    def fit(cls, X, dim, meta=None):
        X = np.asarray(X, np.float64)
        mean, scale = X.mean(0), X.std(0) + 1e-6
        _, S, Vt = np.linalg.svd((X - mean) / scale, full_matrices=False)
        dim = min(dim, len(S))
        std = S[:dim] / math.sqrt(max(len(X) - 1, 1))
        return cls(mean, scale, Vt[:dim], std, meta)

    def standardize(self, X):
        return (np.asarray(X, np.float64) - self.mean) / self.scale

    def transform(self, X):
        return (self.standardize(X) @ self.components.T / self.std).astype(np.float32)

    def explained(self):
        """Fraction of the standardised variance the kept components carry."""
        return float((self.std ** 2).sum() / len(self.mean))

    def save(self, path):
        np.savez(path, mean=self.mean, scale=self.scale, components=self.components,
                 std=self.std, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path):
        f = np.load(path)
        return cls(f["mean"], f["scale"], f["components"], f["std"], json.loads(str(f["meta"])))


def question_text(instance):
    """The question text that is encoded; the same text the environment hands the controller."""
    from longmemeval_bandit import oracle_question
    return oracle_question(instance)


def cosine_rows(projector, A, B):
    """Row-wise cosine of A and B (same shape), in standardised activation space."""
    A, B = projector.standardize(A), projector.standardize(B)
    return ((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
            ).astype(np.float32)


def zscore_within(values, groups=None):
    """Standardise within each group (default: all rows are one context). One context's turns
    are the bandit's candidates, so how much more a turn matches than its neighbours is what
    decides which one to act on; the absolute cosine level drifts from question to question."""
    values = np.asarray(values, np.float32)
    groups = np.zeros(len(values), int) if groups is None else np.asarray(groups)
    out = np.zeros_like(values)
    for g in np.unique(groups):
        m = groups == g
        sd = values[m].std()
        out[m] = (values[m] - values[m].mean()) / sd if sd > 1e-6 else 0
    return out


# [turn | question] representations. Pt, Pq: whitened PCA projections of the turn and of the
# question (one row per turn); cos / cos_z: their cosine, raw and standardised within the context.
#   turn         [Pt]                      question-blind; the only one valid with the oracle off
#   concat       [Pt, Pq]                  the plain 2 x dim stack. A stationary kernel compares
#                                          (turn, question) pairs with other pairs, so it learns
#                                          "turns like this helped questions like this" but is
#                                          never told whether this turn matches this question;
#                                          within one instance Pq is constant and ranks nothing.
#                                          Needs many instances to learn the match itself.
#   cosine       [cos, cos_z]              the match itself, no learning, 2 dims: data-efficient,
#                                          but one number can't say which aspects of a match matter
#                                          and LM activations aren't trained for similarity.
#   product      [Pt*Pq, cos, cos_z]       per-dimension agreement. A weight per dim of Pt*Pq is a
#                                          learned diagonal bilinear similarity sum_i w_i t_i q_i:
#                                          the GP (with ARD) learns which directions of agreement
#                                          predict reward, and that transfers across questions.
#   interaction  [Pt*Pq, Pt, cos, cos_z]   product plus the turn's own content, for the
#   (default)                              question-independent part of a turn's value (stated
#                                          personal facts vs. filler). Pq alone is left out: it is
#                                          constant within an instance and only spends kernel dims.
# All are followed by META_FEATURES when meta=True. Full bilinear t^T W q (dim^2 weights) and
# |Pt - Pq| were considered: the first is too many parameters for bandit-sized data, the second
# carries the same agreement information as Pt*Pq with the sign of each dimension lost.
REPRESENTATIONS = {
    "turn": lambda Pt, Pq, c, cz: [Pt],
    "concat": lambda Pt, Pq, c, cz: [Pt, Pq],
    "cosine": lambda Pt, Pq, c, cz: [c[:, None], cz[:, None]],
    "product": lambda Pt, Pq, c, cz: [Pt * Pq, c[:, None], cz[:, None]],
    "interaction": lambda Pt, Pq, c, cz: [Pt * Pq, Pt, c[:, None], cz[:, None]],
}
DEFAULT_REPRESENTATION = "interaction"


def represent(name, Pt, Pq=None, cos=None, cos_z=None):
    if name != "turn" and Pq is None:
        raise ValueError(f"representation {name!r} needs the question; use 'turn' without it")
    return np.hstack(REPRESENTATIONS[name](Pt, Pq, cos, cos_z)).astype(np.float32)


def representation_names(name, dim):
    pcs = lambda prefix: [f"{prefix}{i}" for i in range(dim)]
    return {"turn": pcs("turn_pc"), "concat": pcs("turn_pc") + pcs("question_pc"),
            "cosine": ["cos", "cos_z"], "product": pcs("agree_pc") + ["cos", "cos_z"],
            "interaction": pcs("agree_pc") + pcs("turn_pc") + ["cos", "cos_z"]}[name]


class Featurizer:
    """Per instance: begin(question) once (one encoder call, or none with the oracle off).
    Per step: __call__(turns, t, T) -> (len(turns), len(feature_names)) GP inputs.

    oracle=False (pair with replay --no-question-oracle) forces the question-blind "turn"
    representation, so the same controller code runs both ways. The turns' "evidence" flag is
    never read: it is the answer key.
    """

    def __init__(self, encoder, projector, representation=DEFAULT_REPRESENTATION, meta=True,
                 oracle=True):
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unknown representation {representation!r}; "
                             f"choose from {', '.join(REPRESENTATIONS)}")
        self.encoder, self.projector, self.meta, self.oracle = encoder, projector, meta, oracle
        self.representation = representation if oracle else "turn"
        self.q_vector = self.Pq = None

    @property
    def feature_names(self):
        return (representation_names(self.representation, len(self.projector.std))
                + (META_FEATURES if self.meta else []))

    def begin(self, question):
        """Call from Controller.begin(question_id, question)."""
        if self.representation != "turn" and question is None:
            raise RuntimeError("no question given: run replay with the oracle on, or build the "
                               "featurizer with oracle=False")
        self.q_vector = None if question is None else self.encoder.encode([question])[0]
        self.Pq = None if question is None else self.projector.transform(self.q_vector[None])[0]

    def __call__(self, turns, t, T):
        n = len(turns)
        V = self.encoder.encode([turn["text"] for turn in turns])
        Pt = self.projector.transform(V)
        if self.representation == "turn":
            X = represent("turn", Pt)
        else:
            cos = cosine_rows(self.projector, V, np.repeat(self.q_vector[None], n, axis=0))
            X = represent(self.representation, Pt, np.repeat(self.Pq[None], n, axis=0),
                          cos, zscore_within(cos))
        if not self.meta:
            return X
        meta = np.array([[
            turn["role"] == "user",
            math.log1p(len(turn["text"]) / 4) / 8,  # ~tokens, log-scaled to about 0..1
            (n - 1 - i) / max(n - 1, 1),            # 0 = newest turn, 1 = oldest
            turn["edited"],
            (T - t) / T,                            # share of frontier calls still to come
        ] for i, turn in enumerate(turns)], np.float32).reshape(n, len(META_FEATURES))
        return np.hstack([X, meta])


def load_featurizer(projector_path, representation=DEFAULT_REPRESENTATION, meta=True,
                    oracle=True, **encoder_kwargs):
    """Featurizer using the model, layer and pooling saved with the projector."""
    projector = Projector.load(projector_path)
    m = projector.meta
    encoder = TurnEncoder(m["model"], [m["layer"]], m["pooling"], m["max_tokens"], **encoder_kwargs)
    return Featurizer(encoder, projector, representation, meta, oracle)


# --- data for probe / fit -----------------------------------------------------------------
def load_turns(path, n_instances, distractors, holdout_limit, seed):
    """(turn rows, {question_id: question text}) from n_instances that a
    `replay --limit holdout_limit --seed seed` run won't use."""
    from longmemeval_bandit import exchanges, select
    with open(path) as f:
        data = json.load(f)
    held = {inst["question_id"] for inst in select(data, holdout_limit, True, seed)} \
        if holdout_limit else set()
    pool = [inst for inst in data if inst["question_id"] not in held]
    rows, questions = [], {}
    for inst in select(pool, n_instances, True, seed + 1):
        questions[inst["question_id"]] = question_text(inst)
        rng = random.Random(f"{seed}:{inst['question_id']}")
        for exchange in exchanges(inst, distractors, rng):
            for turn in exchange:
                rows.append({**turn, "question_id": inst["question_id"],
                             "question_type": inst.get("question_type")})
    return rows, questions


def _metrics(rows, score):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.array([r["evidence"] for r in rows])
    by_qid = {}
    for r, s in zip(rows, score):
        by_qid.setdefault(r["question_id"], []).append((s, r["evidence"]))
    kept = total = 0  # evidence surviving if the lower-scored half of each context were deleted
    for items in by_qid.values():
        items.sort(key=lambda x: -x[0])
        kept += sum(e for _, e in items[:math.ceil(len(items) / 2)])
        total += sum(e for _, e in items)
    return roc_auc_score(y, score), average_precision_score(y, score), kept / max(total, 1)


def run_probe(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows, questions = load_turns(args.data, args.instances, args.distractors,
                                 args.holdout_limit, args.seed)
    y = np.array([r["evidence"] for r in rows])
    groups = [r["question_id"] for r in rows]
    if y.sum() < 5:
        raise SystemExit(f"only {y.sum()} evidence turns; use more --instances")
    print(f"{len(rows)} turns from {len(questions)} instances, {y.sum()} evidence "
          f"({y.mean():.2%} base rate)", file=sys.stderr)
    encoder = TurnEncoder(args.model, args.layers, args.pooling, args.max_tokens,
                          args.batch_size, args.device)
    print(f"{args.model}: {encoder.num_layers} blocks, probing {encoder.layers}", file=sys.stderr)
    qids = list(questions)
    turn_emb = encoder.encode_layers([r["text"] for r in rows], progress=True)
    q_emb = encoder.encode_layers([questions[q] for q in qids])
    q_index = np.array([qids.index(r["question_id"]) for r in rows])
    meta = np.array([[r["role"] == "user", math.log1p(len(r["text"]))] for r in rows])
    cv = GroupKFold(n_splits=min(5, len(qids)))

    def oof(X):
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=args.C, class_weight="balanced", max_iter=3000))
        return cross_val_predict(clf, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]

    def line(name, score):
        auc, ap, kept = _metrics(rows, score)
        print(f"{name:<34}{auc:>8.3f}{ap:>8.3f}{kept:>14.3f}")

    print(f"\n{'features (+ role, length)':<34}{'AUROC':>8}{'AP':>8}{'kept @50%':>14}")
    print(f"{'(chance)':<34}{0.5:>8.3f}{y.mean():>8.3f}{0.5:>14.3f}")
    line("role + length only", oof(meta))
    for layer in encoder.layers:
        # Projector fitted on all probed texts: unsupervised, so no label leakage across folds.
        proj = Projector.fit(np.vstack([turn_emb[layer], q_emb[layer]]), args.dim)
        Pt, Pq = proj.transform(turn_emb[layer]), proj.transform(q_emb[layer])[q_index]
        cos = cosine_rows(proj, turn_emb[layer], q_emb[layer][q_index])
        cos_z = zscore_within(cos, q_index)
        tag = f"L{layer}/{encoder.num_layers}"
        line(f"{tag} cosine, untrained", cos)
        for name in args.representations:
            line(f"{tag} {name}", oof(np.hstack([represent(name, Pt, Pq, cos, cos_z), meta])))
    print(f"\nLinear probes on {args.dim} whitened PCA dims per vector, cross-validated by "
          "instance; \"kept @50%\" = evidence surviving deletion of the lower-scored half of each "
          "context. A linear probe can't use turn x question interactions inside 'concat' (a GP "
          "can, given enough instances), so 'concat' scoring like 'turn' here is expected.")


def run_fit(args):
    rows, questions = load_turns(args.data, args.instances, args.distractors,
                                 args.holdout_limit, args.seed)
    encoder = TurnEncoder(args.model, [args.layer] if args.layer else None, args.pooling,
                          args.max_tokens, args.batch_size, args.device)
    X = encoder.encode([r["text"] for r in rows] + list(questions.values()), progress=True)
    projector = Projector.fit(X, args.dim, meta={
        "model": args.model, "layer": encoder.layers[0], "pooling": args.pooling,
        "max_tokens": args.max_tokens, "dim": args.dim, "fit_turns": len(rows),
    })
    projector.save(args.out)
    print(f"saved {args.out}: layer {encoder.layers[0]}/{encoder.num_layers}, {len(rows)} turns, "
          f"+ {len(questions)} questions, {args.dim} dims keep {projector.explained():.1%} of standardised variance",
          file=sys.stderr)


def run_encode(args):
    texts = args.text or [line.rstrip("\n") for line in sys.stdin if line.strip()]
    encoder = TurnEncoder(args.model, [args.layer] if args.layer else None, args.pooling,
                          args.max_tokens, args.batch_size, args.device)
    X = encoder.encode(texts)
    if args.projector:
        X = Projector.load(args.projector).transform(X)
    if args.out:
        np.save(args.out, X)
        print(f"saved {X.shape} to {args.out}", file=sys.stderr)
    else:
        for vector in X:
            print(json.dumps([round(float(v), 5) for v in vector]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    phases = parser.add_subparsers(dest="phase", required=True)

    def model_args(p):
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--pooling", choices=["mean", "last"], default="mean")
        p.add_argument("--max-tokens", type=int, default=512, help="longer turns are truncated")
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--device", help="cuda, mps or cpu (default: best available)")

    def data_args(p):
        p.add_argument("data")
        p.add_argument("--instances", type=int, default=30)
        p.add_argument("--distractors", type=int, default=8)
        p.add_argument("--holdout-limit", type=int, default=60,
                       help="the --limit of your replay runs; those instances are excluded")
        p.add_argument("--seed", type=int, default=0, help="the --seed of your replay runs")
        p.add_argument("--dim", type=int, default=32, help="PCA dims the GP will see")

    p = phases.add_parser("probe", help="which layer's activations predict evidence turns?")
    data_args(p)
    model_args(p)
    p.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")],
                   default=[4, 8, 12, 14, 16, 18, 20, 24, 28])
    p.add_argument("--C", type=float, default=0.1, help="inverse L2 strength of the probe")
    p.add_argument("--representations", type=lambda v: v.split(","),
                   default=list(REPRESENTATIONS), help="comma-separated subset to compare")
    p = phases.add_parser("fit", help="fit and save the projector for one layer")
    data_args(p)
    model_args(p)
    p.add_argument("--layer", type=int)
    p.add_argument("--out", required=True)
    p = phases.add_parser("encode", help="print activation vectors for text (args or stdin)")
    p.add_argument("text", nargs="*")
    model_args(p)
    p.add_argument("--layer", type=int)
    p.add_argument("--projector", help="output projected features instead of raw vectors")
    p.add_argument("--out", help=".npy file instead of JSON lines on stdout")
    args = parser.parse_args()
    {"probe": run_probe, "fit": run_fit, "encode": run_encode}[args.phase](args)
