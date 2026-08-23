"""Exact-match scoring with minimal, task-agnostic normalization."""
import re
import re as _re

_STRIP = re.compile(r"^[\s`'\"*]+|[\s`'\"*.,!]+$")

_FMT = {
    "t1_add": _re.compile(r"^-?\d+$"),
    "t2_reverse": _re.compile(r"^[a-z]+$"),
    "t3_date": _re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "t4_sentiment": _re.compile(r"^(positive|negative)$"),
    "t5_extract": _re.compile(r"^[a-z]+$"),
}


def normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().split("\n")[0]
    s = _STRIP.sub("", s)
    return s.strip().lower()


def is_correct(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def score_rows(rows, task=None) -> dict:
    n = k = wf = wf_k = 0
    ov = 0.0
    t = task or ""
    for r in rows:
        n += 1
        c = is_correct(r["pred"], r["answer"])
        w = is_wellformed(r["pred"], t or r.get("task", ""))
        k += int(c)
        wf += int(w)
        wf_k += int(c and w)
        ov += char_overlap(r["pred"], r["answer"])
    return {"n": n, "correct": k, "acc": (k / n) if n else None,
            "char_overlap": (ov / n) if n else None,
            "wellformed": (wf / n) if n else None,
            "n_wellformed": wf,
            "acc_given_wellformed": (wf_k / wf) if wf else None}


def char_overlap(pred: str, gold: str) -> float:
    """Positional character accuracy. Secondary metric for floor tasks like reversal."""
    p, g = normalize(pred), normalize(gold)
    if not g:
        return 0.0
    m = sum(1 for i, c in enumerate(g) if i < len(p) and p[i] == c)
    return m / len(g)

def is_wellformed(pred: str, task: str) -> bool:
    """Did the output obey the format spec, regardless of whether it's right?"""
    pat = _FMT.get(task)
    return bool(pat.match(normalize(pred))) if pat else True