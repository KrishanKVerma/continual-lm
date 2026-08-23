"""Generate task streams. Exact-match scorable, no downloads.

Trained tasks: t1_add, t2_reverse, t3_date, t4_sentiment, t5_extract, t3b_date_us
Control tasks (eval only, never trained): c1_multiply, c2_country, c3_wordcount
"""
import json, random, argparse
from pathlib import Path

OUT = Path("data/tasks")


def t1_add(rng):
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    return f"Compute: {a} + {b}", str(a + b)


def t2_reverse(rng):
    w = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(5, 9)))
    return f"Reverse this string: {w}", w[::-1]


def t3_date(rng):
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1990, 2025)
    return f"Convert to YYYY-MM-DD: {d:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{d:02d}"


def t4_sentiment(rng):
    subj = ["The battery", "The screen", "Delivery", "The fabric", "Support", "The app",
            "Setup", "The finish", "Packaging", "The sound", "The build", "The warranty"]
    pos = ["held up perfectly", "exceeded what I expected", "was worth every rupee",
           "impressed me", "worked on the first try", "is genuinely excellent"]
    neg = ["failed within a week", "was a total letdown", "wasted my money",
           "stopped working immediately", "is unusable", "was far worse than described"]
    tail = ["", " Would buy again.", " Not recommended.", " Second purchase.",
            " Returned it.", " Honestly surprised.", " Mixed feelings aside.",
            " Ordered on Monday.", " Shipped from Delhi.", " Used it for a month."]
    s, t = rng.choice(subj), rng.choice(tail)
    if rng.random() < 0.5:
        return f"Sentiment (positive/negative): {s} {rng.choice(pos)}.{t}", "positive"
    return f"Sentiment (positive/negative): {s} {rng.choice(neg)}.{t}", "negative"


def t5_extract(rng):
    names = ["Priya", "Marcus", "Wei", "Sofia", "Arjun", "Lena", "Omar", "Ines",
             "Takeshi", "Nadia", "Diego", "Fatima"]
    cities = ["Berlin", "Osaka", "Nairobi", "Lisbon", "Toronto", "Jaipur",
              "Helsinki", "Bogota", "Cairo", "Seoul", "Dublin", "Perth"]
    verbs = ["flew to", "relocated to", "landed in", "drove to", "was posted to", "arrived in"]
    when = ["last Tuesday", "in March", "over the weekend", "yesterday",
            "two years ago", "on short notice"]
    c = rng.choice(cities)
    return (f"Extract the city name: {rng.choice(names)} {rng.choice(verbs)} "
            f"{c} {rng.choice(when)}.", c)


# ---- Conflicting task: same surface as t3_date, different required output ----
def t3b_date_us(rng):
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1990, 2025)
    return f"Convert to MM/DD/YYYY: {d:02d}/{m:02d}/{y}", f"{m:02d}/{d:02d}/{y}"


# ---- Control tasks: NEVER trained. Measure general capability erosion. ----
def c1_multiply(rng):
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    return f"Compute: {a} * {b}", str(a * b)


def c2_country(rng):
    pairs = [("France", "Paris"), ("Japan", "Tokyo"), ("Brazil", "Brasilia"),
             ("Kenya", "Nairobi"), ("Norway", "Oslo"), ("Egypt", "Cairo"),
             ("Peru", "Lima"), ("Nepal", "Kathmandu"), ("Cuba", "Havana"),
             ("Ghana", "Accra"), ("Iraq", "Baghdad"), ("Chile", "Santiago"),
             ("Poland", "Warsaw"), ("Sweden", "Stockholm"), ("Vietnam", "Hanoi"),
             ("Morocco", "Rabat"), ("Bolivia", "Sucre"), ("Ireland", "Dublin")]
    lead = ["What is the capital of", "Name the capital of", "Capital city of",
            "Which city is the capital of", "State the capital of"]
    hedge = ["", " Answer briefly.", " Just the name.", " Be concise.",
             " One word.", " No explanation.", " Quickly.", " Short answer."]
    c, cap = rng.choice(pairs)
    return f"{rng.choice(lead)} {c}?{rng.choice(hedge)}", cap


def c3_wordcount(rng):
    pool = ["red", "quiet", "iron", "swift", "hollow", "bright", "narrow",
            "stone", "distant", "clear", "sharp", "warm", "pale", "coarse",
            "dense", "faint", "rough", "steady"]
    n = rng.randint(3, 9)
    ws = [rng.choice(pool) for _ in range(n)]
    return f"How many words are in this list: {' '.join(ws)}", str(n)


TASKS = {
    "t1_add": t1_add,
    "t2_reverse": t2_reverse,
    "t3_date": t3_date,
    "t4_sentiment": t4_sentiment,
    "t5_extract": t5_extract,
    "t3b_date_us": t3b_date_us,
    "c1_multiply": c1_multiply,
    "c2_country": c2_country,
    "c3_wordcount": c3_wordcount,
}

CONTROLS = {"c1_multiply", "c2_country", "c3_wordcount"}


def build(name, fn, n_train, n_eval, seed):
    rng = random.Random(seed)
    seen, rows = set(), []
    need = n_train + n_eval
    attempts, cap = 0, need * 100
    while len(rows) < need:
        attempts += 1
        if attempts > cap:
            raise RuntimeError(
                f"{name}: exhausted unique prompts at {len(rows)} rows "
                f"(need {need}). Widen the template space."
            )
        p, a = fn(rng)
        if p in seen:
            continue
        seen.add(p)
        rows.append({"task": name, "prompt": p, "answer": a})
    return rows[:n_train], rows[n_train:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for i, (name, fn) in enumerate(TASKS.items()):
        n_tr = 0 if name in CONTROLS else args.n_train
        tr, ev = build(name, fn, n_tr, args.n_eval, args.seed + i)
        assert not (set(r["prompt"] for r in tr) & set(r["prompt"] for r in ev)), \
            f"{name}: train/eval overlap"
        splits = (("eval", ev),) if name in CONTROLS else (("train", tr), ("eval", ev))
        for split, rows in splits:
            path = OUT / f"{name}.{split}.jsonl"
            with path.open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"{path}  {len(rows)}")


if __name__ == "__main__":
    main()