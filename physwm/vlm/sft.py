"""LoRA SFT of Qwen3-VL on Task A + Task B with the default next-token cross-entropy, loss on the
answer tokens only. Vision encoder frozen; LoRA on the language model's attention + MLP, plus
the OneCanvas-style per-patch xyz PE (``physwm.vlm.pe``) trained at 50x the base learning rate.

    python -m physwm.vlm.sft --config configs/sft_qwen3vl2b.yaml [--limit 100 --epochs 30 --out ...]

Frames/views: ``--frames 0,1,2 --views 0`` (or the optional config keys ``frames``/``views``)
put one image per (frame, view) in the user turn, frame-major; each image contributes
``t*(h/2)*(w/2)`` xyz rows from its own depth file, concatenated in the same order as the
processor's ``image_grid_thw`` and therefore as the vision tokens. Missing depth files
(legacy episodes only store t = 0) fall back to zeros for that image. ``--style box3d``
switches prompt and target to centre + extents; ``--no-pe`` trains the B5 ablation.
Task N always targets the first stored frame after the last input frame, so 1-frame arms get
t = 0.1 s and 3-frame arms get t = 0.3 s on the standard frame grid; episodes whose inputs
cover every stored frame are skipped for task N.

M0 (research/PREREG.md): task I (inverse dynamics) always reads the fixed pre-action +
t = 0.1 s frames regardless of ``--frames`` and is skipped for episodes whose grid or
stored states lack that time; ``--style ctx3d`` trains the physical-context arm (per-object
support + frozen action_context payload + future boxes); ``--max-steps`` sets an *exact*
optimizer-step target (caps longer runs, extends shorter ones) so arms can be compared at
matched steps, and ``--plan`` prints per-task sample counts plus the natural/pinned step
count without loading a model -- pin the shared target via --plan before training any arm.
The train log records cumulative padded, unpadded and supervised tokens, so exposure is
auditable and token-matched secondary comparisons are possible without re-running.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from physwm.eval.metrics import NEXT_FRAME_TASK, task_time
from physwm.schema import load_split
from physwm.vlm.pe import (PE_LR_SCALE, frame_token_xyz, image_grid_rows, install_pe,
                           pe_param_groups, save_pe)
from physwm.vlm.prompts import (IDP_TASK, IDP_TIME, build_messages, idp_frame, input_images,
                                prompt_text, target_text)

ASSISTANT_HEADER = "<|im_start|>assistant\n"


def parse_indices(value, default):
    """``"0,2"`` / ``[0, 2]`` / ``None`` -> list of ints (None keeps ``default``)."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [int(v) for v in value.split(",") if v.strip()]
    return [int(v) for v in value]


class SFTData(Dataset):
    def __init__(self, items, tasks, frames=(0,), views=(0,), style: str = "center",
                 processor=None, include_action: bool = True):
        self.frames = list(frames)
        self.views = list(views)
        self.pairs = [(f, v) for f in self.frames for v in self.views]
        self.style = style
        self.include_action = include_action
        self.processor = processor
        self.merge = int(processor.image_processor.merge_size) if processor is not None else 2
        self._grid_cache: dict = {}
        # Task N needs a stored frame after the last input frame; drop episodes whose inputs
        # already cover every stored frame instead of failing the run.
        self.samples = []
        for d, ep in items:
            for t in tasks:
                if t == NEXT_FRAME_TASK:
                    try:
                        task_time(ep, t, self.frames)
                    except ValueError:
                        continue
                if t == IDP_TASK:
                    try:  # fixed t = 0.1 s pair: needs the frame grid and stored states
                        idp_frame(ep)
                        ep.state_at(IDP_TIME)
                    except (KeyError, ValueError):
                        continue
                self.samples.append((d, ep, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        d, ep, task = self.samples[i]
        # Task I always reads the fixed pre-action + early post-action frames, regardless of
        # the arm's --frames setting (research/PREREG.md: fixed times, so no strength leak).
        frames = (0, idp_frame(ep)) if task == IDP_TASK else self.frames
        pairs = [(f, v) for f in frames for v in self.views]
        images = input_images(d, ep, frames, self.views)
        grids = (image_grid_rows(self.processor, images, self._grid_cache)
                 if images else [])
        per_image = [
            frame_token_xyz(d, ep, frame=f, view=v, grid_thw_row=grid, merge=self.merge)
            for (f, v), grid in zip(pairs, grids)
        ]
        xyz = torch.cat(per_image) if per_image else torch.zeros(0, 3)
        return {
            "task": task,
            "messages": build_messages(
                prompt_text(ep, task, self.style, frames, include_action=self.include_action),
                images,
                target_text(ep, task, self.style, frames),
            ),
            "xyz": xyz,
            "grid": grids,
        }


class Collator:
    def __init__(self, processor):
        self.processor = processor
        self.header = processor.tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
        self.merge = int(processor.image_processor.merge_size)

    def _answer_start(self, ids: list[int]) -> int:
        h = self.header
        for s in range(len(ids) - len(h), -1, -1):
            if ids[s:s + len(h)] == h:
                return s + len(h)
        raise ValueError("assistant header not found")

    def _check_alignment(self, enc, batch) -> torch.Tensor:
        """xyz rows must equal the concatenated merged-token count in image order."""
        grid = enc.get("image_grid_thw")
        expected = ([] if grid is None else
                    [[int(t), int(h), int(w)] for t, h, w in grid.tolist()])
        seen = [row for s in batch for row in s["grid"]]
        if expected != seen:
            raise ValueError(f"image order/grid mismatch: processor {expected} vs dataset {seen}")
        tokens = sum(t * (h // self.merge) * (w // self.merge) for t, h, w in expected)
        xyz = torch.cat([s["xyz"] for s in batch]) if batch else torch.zeros(0, 3)
        if xyz.shape[0] != tokens:
            raise ValueError(f"xyz/vision-token mismatch: {xyz.shape[0]} xyz rows for {tokens} tokens")
        return xyz

    def __call__(self, batch):
        enc = self.processor.apply_chat_template(
            [s["messages"] for s in batch], tokenize=True, return_dict=True, return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for r in range(labels.shape[0]):
            labels[r, : self._answer_start(enc["input_ids"][r].tolist())] = -100
        enc["labels"] = labels
        enc["physwm_xyz"] = self._check_alignment(enc, batch).float()
        enc["physwm_tasks"] = [sample["task"] for sample in batch]
        return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N train episodes")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-split", default=None)
    ap.add_argument("--expect-action-types", default=None,
                    help="require exactly these comma-separated episode action types before planning or training")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--frames", default=None, help="comma list of frame indices, e.g. 0,1,2")
    ap.add_argument("--views", default=None, help="comma list of view indices, e.g. 0")
    ap.add_argument("--style", default=None, choices=["center", "box3d", "chain3d", "ctx3d"])
    ap.add_argument("--no-pe", action="store_true", help="train without the xyz PE (B5 ablation)")
    ap.add_argument("--no-action", action="store_true", help="omit action text from prompt (infer from frames)")
    ap.add_argument("--data", default=None, help="override the data root from the config")
    ap.add_argument("--tasks", default=None, help="comma list of tasks, e.g. A,B,N")
    ap.add_argument("--num-workers", type=int, default=None, help="override config num_workers")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="exact optimizer-step target for matched-steps arm comparisons "
                         "(extends shorter arms, caps longer ones)")
    ap.add_argument("--plan", action="store_true",
                    help="print per-task sample counts and the natural/pinned optimizer "
                         "steps, then exit (no model, no GPU)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.data is not None:
        cfg["data"] = args.data
    if args.tasks is not None:
        cfg["tasks"] = [t.strip() for t in args.tasks.split(",")]
    if args.out is not None:
        cfg["out"] = args.out
    if args.train_split is not None:
        cfg["train_split"] = args.train_split
    if args.expect_action_types is not None:
        cfg["expect_action_types"] = [t.strip() for t in args.expect_action_types.split(",")]
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    frames = parse_indices(args.frames if args.frames is not None else cfg.get("frames"), [0])
    views = parse_indices(args.views if args.views is not None else cfg.get("views"), [0])
    style = args.style or cfg.get("style", "center")
    use_pe = not args.no_pe
    include_action = False if args.no_action else bool(cfg.get("include_action", True))
    cfg.update(frames=frames, views=views, style=style, pe=use_pe, include_action=include_action)
    items = load_split(cfg["data"], cfg["train_split"])[: args.limit]
    action_counts = Counter(ep.action.type for _, ep in items)
    if "expect_action_types" in cfg:
        expected = set(cfg["expect_action_types"])
        if set(action_counts) != expected:
            raise ValueError(f"training action types {dict(action_counts)} do not match {sorted(expected)}")
    if args.plan:
        # Matched-steps pinning (research/PREREG.md): the launcher plans every arm before
        # training any of them and pins the shared step target from these counts. No model,
        # no GPU, nothing written.
        ds = SFTData(items, cfg["tasks"], frames, views, style, None,
                     include_action=include_action)
        if not len(ds):
            raise ValueError("training dataset is empty")
        counts: dict = {}
        for _, _, t in ds.samples:
            counts[t] = counts.get(t, 0) + 1
        accum = cfg["grad_accum"]
        # Match the loader's drop_last rule, including a single short batch on tiny sets.
        batches = max(1, len(ds) // cfg["batch_size"])
        natural = max(1, math.ceil(batches * cfg["epochs"] / accum))
        print(json.dumps({
            "plan": True, "data": cfg["data"], "train_split": cfg["train_split"],
            "tasks": cfg["tasks"], "style": style, "frames": frames, "views": views,
            "samples_per_task": counts, "samples_total": len(ds),
            "episodes_per_action_type": dict(action_counts),
            "batch_size": cfg["batch_size"], "grad_accum": accum, "epochs": cfg["epochs"],
            "batches_per_epoch": batches,
            "natural_optimizer_steps": natural,
            "pinned_optimizer_steps": int(cfg["max_steps"]) if cfg.get("max_steps") else None,
        }, indent=1))
        return
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
    pe = None
    if use_pe:
        pe, _hook = install_pe(model)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    if pe is not None:
        pe.float().to("cuda")
        print(f"xyz PE: hidden={pe.hidden}, lr={cfg['lr'] * PE_LR_SCALE:.1e} (50x base)")

    ds = SFTData(items, cfg["tasks"], frames, views, style, processor, include_action=include_action)
    if not len(ds):
        raise ValueError("training dataset is empty")
    collate = Collator(processor)
    print("example target-masked sample:\n", processor.tokenizer.decode(
        [t for t in collate([ds[0]])["labels"][0].tolist() if t != -100]))
    nw = cfg["num_workers"] if args.num_workers is None else args.num_workers
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=nw,
                    collate_fn=collate, drop_last=len(ds) >= cfg["batch_size"],
                    persistent_workers=nw > 0)

    accum = cfg["grad_accum"]
    total_steps = max(1, math.ceil(len(dl) * cfg["epochs"] / accum))
    if cfg.get("max_steps"):
        # Exact target (matched-steps rule, research/PREREG.md): a count below the natural
        # schedule caps the run, one above extends it. Pin the shared count via --plan
        # before training any arm; the launcher pins the max natural count across arms.
        want = max(1, int(cfg["max_steps"]))
        if want > total_steps:
            print(f"extending natural schedule ({total_steps} steps) to the pinned {want}")
        total_steps = want
    warmup = int(cfg["warmup_frac"] * total_steps)
    base = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": base}] + (pe_param_groups(pe, cfg["lr"]) if pe is not None else [])
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    counts = {}
    for _, _, t in ds.samples:
        counts[t] = counts.get(t, 0) + 1
    print(f"{len(ds)} samples {counts}, {len(dl)} batches/epoch, {total_steps} optimizer steps")

    model.train()
    step, micro = 0, 0
    tokens = tokens_in = tokens_sup = 0
    seen_tasks = Counter({task: 0 for task in cfg["tasks"]})
    t0, log = time.time(), open(out / "train_log.jsonl", "a")
    running = []
    while step < total_steps:
        for batch in dl:
            seen_tasks.update(batch.pop("physwm_tasks"))
            batch = {k: v.to("cuda") for k, v in batch.items()}
            tokens += int(batch["input_ids"].numel())               # padded input tokens
            tokens_in += int(batch["attention_mask"].sum().item())  # unpadded input
            tokens_sup += int((batch["labels"] != -100).sum().item())  # supervised
            xyz = batch.pop("physwm_xyz", None)
            if pe is not None:
                model._physwm_pe_xyz = xyz  # consumed by the merger hook
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / accum
            loss.backward()
            running.append(loss.item() * accum)
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad]
                + (list(pe.parameters()) if pe is not None else []), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg["log_every"] == 0 or step == total_steps:
                rec = {"step": step, "loss": float(np.mean(running)), "lr": sched.get_last_lr()[0],
                       "elapsed_s": round(time.time() - t0), "tokens": tokens,
                       "tokens_in": tokens_in, "tokens_sup": tokens_sup,
                       "examples_per_task": dict(seen_tasks)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                running = []
            if step % cfg["save_every"] == 0:
                ckpt = out / f"checkpoint-{step}"
                model.save_pretrained(ckpt)
                if pe is not None:
                    save_pe(ckpt / "pe.pt", pe)
            if step >= total_steps:
                break
    model.save_pretrained(out / "adapter")
    if pe is not None:
        save_pe(out / "pe.pt", pe)
        print("saved", out / "pe.pt")
    print("saved", out / "adapter")


if __name__ == "__main__":
    main()
