"""LoRA fine-tune on a single task. Loss masked to answer tokens only."""
import json, argparse, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

TASKS_DIR = Path("data/tasks")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class TaskData(Dataset):
    def __init__(self, task, tok, max_len, n=None):
        self.rows = [json.loads(l) for l in (TASKS_DIR / f"{task}.train.jsonl").open()]
        if n:
            self.rows = self.rows[:n]
        self.tok, self.max_len = tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        msgs = [{"role": "user", "content": r["prompt"] + "\nAnswer with only the result."}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(r["answer"] + self.tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + a_ids)[: self.max_len]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids, labs, mask = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        labs.append(b["labels"] + [-100] * k)
        mask.append([1] * len(b["input_ids"]) + [0] * k)
    return (torch.tensor(ids), torch.tensor(labs), torch.tensor(mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--task", required=True)
    ap.add_argument("--resume_adapter", default=None, help="continue from a previous task's adapter")
    ap.add_argument("--out", required=True, help="adapter output dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--lora_r", type=int, default=16)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ))
    model.print_trainable_parameters()
    model.train()

    ds = TaskData(args.task, tok, args.max_len, args.n_train)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total = len(dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * total), total)

    step = 0
    for ep in range(args.epochs):
        for ids, labs, mask in dl:
            ids, labs, mask = ids.cuda(), labs.cuda(), mask.cuda()
            loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"ep{ep} step{step}/{total} loss {loss.item():.4f}", flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    json.dump({"task": args.task, "seed": args.seed, "model_id": args.model_id,
               "resume_adapter": args.resume_adapter, "epochs": args.epochs,
               "lr": args.lr, "n_train": len(ds), "lora_r": args.lora_r,
               "final_loss": float(loss.item())},
              open(Path(args.out) / "run_meta.json", "w"), indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()