"""Assemble the forgetting matrix from a run's step summaries."""
import json, argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--metric", default="acc",
                    choices=["acc", "acc_given_wellformed", "wellformed", "char_overlap"])
    ap.add_argument("--baseline", default="base_zeroshot")
    args = ap.parse_args()

    man = json.load(open(Path("results", args.run, "manifest.json")))
    order = man["order"]

    rows = []
    base = json.load(open(Path("results", args.baseline, "summary.json")))["tasks"]
    rows.append(("base", [base[t][args.metric] for t in order]))

    for i, task in enumerate(order, 1):
        s = json.load(open(Path("results", args.run, f"step{i}_{task}", "summary.json")))["tasks"]
        rows.append((f"after_{task}", [s[t][args.metric] for t in order]))

    w = max(len(r[0]) for r in rows) + 2
    print(f"metric: {args.metric}   run: {args.run}   seed: {man['seed']}\n")
    print(" " * w + "".join(f"{t:>16s}" for t in order))
    for name, vals in rows:
        print(f"{name:<{w}s}" + "".join(f"{v:>16.3f}" for v in vals))

    final = rows[-1][1]
    print("\nretention vs each task's own peak:")
    for j, t in enumerate(order):
        peak = max(r[1][j] for r in rows)
        print(f"  {t:<14s} peak={peak:.3f}  final={final[j]:.3f}  drop={peak - final[j]:+.3f}")

    out = Path("results", args.run, f"matrix_{args.metric}.json")
    out.write_text(json.dumps({"order": order, "metric": args.metric,
                               "rows": {n: v for n, v in rows}}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()