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
    n = 0
    k = 0
    for r in rows:
        n += 1
        k += int(is_correct(r["pred"], r["answer"]))
    return {"n": n, "correct": k, "acc": (k / n) if n else None}