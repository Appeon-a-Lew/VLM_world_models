"""Batched Qwen3-VL inference (base model or + LoRA adapter) -> predictions.jsonl.

    python -m physwm.vlm.infer --data $WM/data --splits id_test --tasks A B \
        --out $WM/runs/zs/pred.jsonl [--lora $WM/runs/sft/adapter] [--image-mode swap] [--style qwen3d]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from physwm.schema import load_split
from physwm.vlm.parse import parse_answer
from physwm.vlm.prompts import build_messages, load_image, prompt_text, swap_partners


def load_model(model_id: str, lora: str | None = None):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    )
    if lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora)
        model = model.merge_and_unload()
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


@torch.inference_mode()
def generate(model, processor, batch_msgs: list[list[dict]], max_new_tokens: int) -> list[str]:
    inputs = processor.apply_chat_template(
        batch_msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", processor_kwargs={"padding": True},
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=["A", "B"])
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--style", default="center", choices=["center", "qwen3d"])
    ap.add_argument("--image-mode", default="normal", choices=["normal", "none", "blank", "swap"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="episodes per split")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, processor = load_model(args.model, args.lora)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start, n_done = time.time(), 0
    with open(out_path, "w") as f:
        for split in args.splits:
            items = load_split(args.data, split)[: args.limit]
            partners = swap_partners(items) if args.image_mode == "swap" else {}
            work = [(d, ep, task) for d, ep in items for task in args.tasks]
            for i in tqdm(range(0, len(work), args.batch_size), desc=split):
                chunk = work[i:i + args.batch_size]
                msgs = [
                    build_messages(
                        prompt_text(ep, task, args.style),
                        load_image(d, ep, args.image_mode, partners.get(ep.episode_id)),
                    )
                    for d, ep, task in chunk
                ]
                per_object = 128 if args.style == "qwen3d" else 48
                max_new = 48 + per_object * max(len(ep.objects) for _, ep, _ in chunk)
                answers = generate(model, processor, msgs, max_new)
                for (d, ep, task), ans in zip(chunk, answers):
                    objects, parsed = parse_answer(ans, ep)
                    f.write(json.dumps({
                        "episode_id": ep.episode_id, "task": task, "objects": objects,
                        "parsed": parsed, "raw": ans, "image_mode": args.image_mode,
                        "style": args.style, "lora": args.lora,
                    }) + "\n")
                n_done += len(chunk)
    print(f"wrote {out_path} ({n_done} samples, {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
