"""LoRA SFT of Qwen3-VL on Task A + Task B with the default next-token cross-entropy, loss on the
answer tokens only. Vision encoder frozen; LoRA on the language model's attention + MLP.

    python -m physwm.vlm.sft --config configs/sft_qwen3vl2b.yaml [--limit 100 --epochs 30 --out ...]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from physwm.schema import load_split
from physwm.vlm.prompts import build_messages, load_image, prompt_text, target_text

ASSISTANT_HEADER = "<|im_start|>assistant\n"


class SFTData(Dataset):
    def __init__(self, items, tasks):
        self.samples = [(d, ep, t) for d, ep in items for t in tasks]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        d, ep, task = self.samples[i]
        return build_messages(prompt_text(ep, task, "center"), load_image(d, ep), target_text(ep, task))


class Collator:
    def __init__(self, processor):
        self.processor = processor
        self.header = processor.tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)

    def _answer_start(self, ids: list[int]) -> int:
        h = self.header
        for s in range(len(ids) - len(h), -1, -1):
            if ids[s:s + len(h)] == h:
                return s + len(h)
        raise ValueError("assistant header not found")

    def __call__(self, batch):
        enc = self.processor.apply_chat_template(
            batch, tokenize=True, return_dict=True, return_tensors="pt", processor_kwargs={"padding": True}
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for r in range(labels.shape[0]):
            labels[r, : self._answer_start(enc["input_ids"][r].tolist())] = -100
        enc["labels"] = labels
        return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N train episodes")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-split", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.out is not None:
        cfg["out"] = args.out
    if args.train_split is not None:
        cfg["train_split"] = args.train_split
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    processor = AutoProcessor.from_pretrained(cfg["model"])
    processor.tokenizer.padding_side = "right"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    )
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lcfg = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], task_type="CAUSAL_LM",
    ))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    items = load_split(cfg["data"], cfg["train_split"])[: args.limit]
    ds = SFTData(items, cfg["tasks"])
    if not len(ds):
        raise ValueError("training dataset is empty")
    collate = Collator(processor)
    print("example target-masked sample:\n", processor.tokenizer.decode(
        [t for t in collate([ds[0]])["labels"][0].tolist() if t != -100]))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                    collate_fn=collate, drop_last=len(ds) >= cfg["batch_size"],
                    persistent_workers=cfg["num_workers"] > 0)

    accum = cfg["grad_accum"]
    total_steps = max(1, math.ceil(len(dl) * cfg["epochs"] / accum))
    warmup = int(cfg["warmup_frac"] * total_steps)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    print(f"{len(ds)} samples, {len(dl)} batches/epoch, {total_steps} optimizer steps")

    model.train()
    step, micro, t0, log = 0, 0, time.time(), open(out / "train_log.jsonl", "a")
    running = []
    while step < total_steps:
        for batch in dl:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / accum
            loss.backward()
            running.append(loss.item() * accum)
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg["log_every"] == 0 or step == total_steps:
                rec = {"step": step, "loss": float(np.mean(running)), "lr": sched.get_last_lr()[0],
                       "elapsed_s": round(time.time() - t0)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                running = []
            if step % cfg["save_every"] == 0:
                model.save_pretrained(out / f"checkpoint-{step}")
            if step >= total_steps:
                break
    model.save_pretrained(out / "adapter")
    print("saved", out / "adapter")


if __name__ == "__main__":
    main()
