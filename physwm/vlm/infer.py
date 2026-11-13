"""Batched Qwen3-VL inference (base model or + LoRA adapter) -> predictions.jsonl.

    python -m physwm.vlm.infer --data $WM/data --splits id_test --tasks A B \
        --out $WM/runs/zs/pred.jsonl [--lora $WM/runs/sft/adapter] [--image-mode swap] [--style qwen3d]

Phase-2 options: ``--style box3d`` (centre + extents answers), ``--frames 0,1,2 --views 0``
(one image per frame/view, frame-major) and the xyz PE saved as ``pe.pt`` next to the LoRA
adapter (loaded automatically, ``--no-pe`` for the B5 ablation). Each generate call receives
``model._physwm_pe_xyz`` built from the same depth files and image order. With
``--image-mode none|blank`` the PE is skipped (there is no geometry to supply); with
``swap`` it uses the swap partner's depth, so the geometry matches the shown image.
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
from physwm.vlm.parse import parse_answer, parse_box3d
from physwm.vlm.pe import frame_token_xyz, image_grid_rows, install_pe, load_pe
from physwm.vlm.prompts import build_messages, input_images, prompt_text, swap_partners


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


def load_pe_for(model, lora: str | None, enabled: bool):
    """Attach ``pe.pt`` stored next to the adapter; None when absent or disabled (B5)."""
    path = Path(lora) / "pe.pt" if lora and enabled else None
    if path is None or not path.exists():
        return None
    pe = load_pe(path).float()
    install_pe(model, module=pe)
    pe.to(model.device)
    print(f"loaded xyz PE {path} (hidden={pe.hidden})")
    return pe


@torch.inference_mode()
def generate(model, processor, batch_msgs: list[list[dict]], max_new_tokens: int) -> list[str]:
    inputs = processor.apply_chat_template(
        batch_msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", processor_kwargs={"padding": True},
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def parse_indices(value, default=(0,)) -> list[int]:
    if value is None:
        return list(default)
    return [int(v) for v in str(value).split(",") if v.strip()]


def token_budget(style: str, n_objects: int) -> int:
    if style == "box3d":
        return 64 + 40 * n_objects
    per_object = 128 if style == "qwen3d" else 48
    return 48 + per_object * n_objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=["A", "B"])
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--style", default="center", choices=["center", "qwen3d", "box3d"])
    ap.add_argument("--image-mode", default="normal", choices=["normal", "none", "blank", "swap"])
    ap.add_argument("--frames", default=None, help="comma list of frame indices, e.g. 0,1,2")
    ap.add_argument("--views", default=None, help="comma list of view indices, e.g. 0")
    ap.add_argument("--no-pe", action="store_true", help="ignore pe.pt next to the adapter (B5)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="episodes per split")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames, views = parse_indices(args.frames), parse_indices(args.views)
    pairs = [(fr, vw) for fr in frames for vw in views]
    model, processor = load_model(args.model, args.lora)
    pe = load_pe_for(model, args.lora, enabled=not args.no_pe)
    grid_cache: dict = {}
    use_pe = pe is not None and args.image_mode in ("normal", "swap")
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
                msgs, xyz = [], []
                for d, ep, task in chunk:
                    partner = partners.get(ep.episode_id)
                    images = input_images(d, ep, frames, views, args.image_mode, partner)
                    msgs.append(build_messages(prompt_text(ep, task, args.style), images))
                    if use_pe:
                        d_depth, ep_depth = partner if partner is not None else (d, ep)
                        grids = image_grid_rows(processor, images, grid_cache)
                        xyz.append(torch.cat([
                            frame_token_xyz(d_depth, ep_depth, frame=fr, view=vw, grid_thw_row=g)
                            for (fr, vw), g in zip(pairs, grids)
                        ]))
                if pe is not None:
                    model._physwm_pe_xyz = torch.cat(xyz) if xyz else None
                max_new = token_budget(args.style, max(len(ep.objects) for _, ep, _ in chunk))
                answers = generate(model, processor, msgs, max_new)
                for (d, ep, task), ans in zip(chunk, answers):
                    if args.style == "box3d":
                        objects, extents, parsed = parse_box3d(ans, ep)
                    else:
                        objects, parsed = parse_answer(ans, ep)
                        extents = None
                    rec = {
                        "episode_id": ep.episode_id, "task": task, "objects": objects,
                        "parsed": parsed, "raw": ans, "image_mode": args.image_mode,
                        "style": args.style, "lora": args.lora,
                    }
                    if extents is not None:
                        rec["extents"] = extents
                    f.write(json.dumps(rec) + "\n")
                n_done += len(chunk)
    print(f"wrote {out_path} ({n_done} samples, {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
