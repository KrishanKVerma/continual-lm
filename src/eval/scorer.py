"""Exact-match scoring with minimal, task-agnostic normalization."""
import re

_STRIP = re.compile(r"^[\s`'\"*]+|[\s`'\"*.,!]+$")


def normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().split("\n")[0]
    s = _STRIP.sub("", s)
    return s.strip().lower()


def is_correct(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def score_rows(rows) -> dict:
    """rows: iterable of dicts with 'pred' and 'answer'. Returns counts, not just a rate."""
    n = k = 0
    ov = 0.0
    for r in rows:
        n += 1
        k += int(is_correct(r["pred"], r["answer"]))
        ov += char_overlap(r["pred"], r["answer"])
    return {"n": n, "correct": k, "acc": (k / n) if n else None,
            "char_overlap": (ov / n) if n else None}


def char_overlap(pred: str, gold: str) -> float:
    """Positional character accuracy. Secondary metric for floor tasks like reversal."""
    p, g = normalize(pred), normalize(gold)
    if not g:
        return 0.0
    m = sum(1 for i, c in enumerate(g) if i < len(p) and p[i] == c)
    return m / len(g)