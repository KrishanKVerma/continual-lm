"""Generate 5 synthetic task streams. Exact-match scorable, no downloads."""
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


TASKS = {
    "t1_add": t1_add,
    "t2_reverse": t2_reverse,
    "t3_date": t3_date,
    "t4_sentiment": t4_sentiment,
    "t5_extract": t5_extract,
}


def build(name, fn, n_train, n_eval, seed):
    rng = random.Random(seed)
    seen, rows = set(), []
    attempts, cap = 0, (n_train + n_eval) * 100
    while len(rows) < n_train + n_eval:
        attempts += 1
        if attempts > cap:
            raise RuntimeError(
                f"{name}: exhausted unique prompts at {len(rows)} rows "
                f"(need {n_train + n_eval}). Widen the template space."
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
        tr, ev = build(name, fn, args.n_train, args.n_eval, args.seed + i)
        assert not (set(r["prompt"] for r in tr) & set(r["prompt"] for r in ev)), \
            f"{name}: train/eval overlap"
        for split, rows in (("train", tr), ("eval", ev)):
            path = OUT / f"{name}.{split}.jsonl"
            with path.open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"{path}  {len(rows)}")


if __name__ == "__main__":
    main()