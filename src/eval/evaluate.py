"""Evaluate a model on all tasks. Writes per-task counts + per-example predictions."""
import json, argparse, random
from pathlib import Path
from src.eval.scorer import score_rows

TASKS_DIR = Path("data/tasks")


def load_eval(task, n):
    rows = [json.loads(l) for l in (TASKS_DIR / f"{task}.eval.jsonl").open()]
    return rows[:n]


def build_prompt(tok, prompt):
    msgs = [{"role": "user", "content": prompt + "\nAnswer with only the result."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_model(model_id, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def generate(model, tok, prompts, max_new_tokens, batch_size=16):
    import torch
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(model.device)
        with torch.no_grad():
                        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 top_k=None, pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", nargs="+",
                    default=["t1_add", "t2_reverse", "t3_date", "t4_sentiment", "t5_extract"])
    ap.add_argument("--n_per_task", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--tag", required=True, help="e.g. after_t1_seed0_naive")
    ap.add_argument("--stub", action="store_true", help="no model; random preds. plumbing test only")
    args = ap.parse_args()

    model = tok = None
    if not args.stub:
        model, tok = load_model(args.model_id, args.adapter)

    out_dir = Path("results") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"tag": args.tag, "model_id": args.model_id, "adapter": args.adapter,
               "stub": args.stub, "tasks": {}}

    for task in args.tasks:
        rows = load_eval(task, args.n_per_task)
        if args.stub:
            rng = random.Random(hash(task) % 10_000)
            preds = [r["answer"] if rng.random() < 0.5 else "wrong" for r in rows]
        else:
            preds = generate(model, tok,
                             [build_prompt(tok, r["prompt"]) for r in rows],
                             args.max_new_tokens)
        for r, p in zip(rows, preds):
            r["pred"] = p
        summary["tasks"][task] = score_rows(rows)
        with (out_dir / f"{task}.preds.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        s = summary["tasks"][task]
        print(f"{task}: {s['correct']}/{s['n']} = {s['acc']:.3f}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()