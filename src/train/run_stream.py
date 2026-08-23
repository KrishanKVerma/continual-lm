"""Sequential task stream. Trains one adapter across tasks, evals all tasks after each."""
import argparse, subprocess, sys, json
from pathlib import Path


def sh(cmd):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", nargs="+",
                    default=["t1_add", "t2_reverse", "t3_date", "t4_sentiment", "t5_extract"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", required=True, help="run name, e.g. naive_seed0")
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_per_task", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    prev = None
    for i, task in enumerate(args.order, 1):
        out = f"adapters/{args.run}/step{i}_{task}"
        if Path(out, "adapter_config.json").exists():
            print(f"skip train step{i} ({task}) — adapter exists", flush=True)
        else:
            cmd = ["python", "-m", "src.train.train_task", "--task", task, "--out", out,
                   "--seed", str(args.seed), "--n_train", str(args.n_train),
                   "--epochs", str(args.epochs)]
            if prev:
                cmd += ["--resume_adapter", prev]
            sh(cmd)
        tag = f"{args.run}/step{i}_{task}"
        if Path("results", tag, "summary.json").exists():
            print(f"skip eval step{i} — summary exists", flush=True)
        else:
            sh(["python", "-m", "src.eval.evaluate", "--adapter", out, "--tag", tag,
                "--n_per_task", str(args.n_per_task), "--tasks", *args.order])
        prev = out

    manifest = {"run": args.run, "seed": args.seed, "order": args.order,
                "n_train": args.n_train, "n_per_task": args.n_per_task,
                "epochs": args.epochs}
    Path("results", args.run).mkdir(parents=True, exist_ok=True)
    Path("results", args.run, "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("stream complete:", args.run)


if __name__ == "__main__":
    main()