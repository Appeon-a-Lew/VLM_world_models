"""Batched Qwen3-VL inference (base model or + LoRA adapter) -> predictions.jsonl.

    python -m physwm.vlm.infer --data $WM/data --splits id_test --tasks A B \
        --out $WM/runs/zs/pred.jsonl [--lora $WM/runs/sft/adapter] [--image-mode swap] [--style qwen3d]

Phase-2 options: ``--style box3d`` (centre + extents answers), ``--frames 0,1,2 --views 0``
(one image per frame/view, frame-major) and the xyz PE saved as ``pe.pt`` next to the LoRA
adapter (loaded automatically, ``--no-pe`` for the B5 ablation). Each generate call receives
``model._physwm_pe_xyz`` built from the same depth files and image order. With
``--image-mode none|blank`` the PE is skipped (there is no geometry to supply); with
``swap`` it uses the swap partner's depth, so the geometry matches the shown image.

Every record carries ``"frames"`` (the input frame indices) so task N is scored at the first
stored frame after the last input frame without re-deriving the arm shape from the file name.

M0 options (research/PREREG.md): ``--style ctx3d`` (emit the physical context before the
future boxes; the emitted ``action_context`` is kept in the record for the predicted-context
diagnostic), ``--tasks ... I`` (inverse-dynamics answers on the fixed t = 0.1 s input pair,
parsed into the record's ``"idp"`` field) and ``--oracle-context`` (inject the
simulator-provided action context into forward-prediction prompts; task A never gets it).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from physwm.eval.metrics import NEXT_FRAME_TASK
from physwm.schema import Episode, load_split
from physwm.vlm.parse import parse_answer, parse_box3d, parse_chain3d, parse_ctx3d, parse_idp
from physwm.vlm.pe import frame_token_xyz, image_grid_rows, install_pe, load_pe
from physwm.vlm.prompts import (IDP_TASK, IDP_TIME, build_messages, idp_frame, input_images,
                                prompt_text, swap_partners)


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
    """Attach ``pe.pt`` saved with the adapter; None when absent or disabled (B5).

    ``sft.py`` writes ``<out>/adapter`` and ``<out>/pe.pt``, so ``--lora <out>/adapter``
    places ``pe.pt`` in ``Path(lora).parent``. Checks both ``Path(lora) / "pe.pt"`` and
    ``Path(lora).parent / "pe.pt"``, and fails loudly if ``config.json`` says ``pe=true``
    but ``pe.pt`` is missing.
    """
    if not lora or not enabled:
        return None
    lora_path = Path(lora)
    candidates = [lora_path / "pe.pt", lora_path.parent / "pe.pt"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        for cfg_path in (lora_path / "config.json", lora_path.parent / "config.json"):
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text())
                if cfg.get("pe"):
                    raise FileNotFoundError(
                        f"training config {cfg_path} has pe=true, but pe.pt was not found in {candidates}"
                    )
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
    if style == "ctx3d":
        return 192 + 160 * n_objects
    if style == "chain3d":
        return 96 + 128 * n_objects
    if style == "box3d":
        return 64 + 64 * n_objects
    if style == "qwen3d":
        return 48 + 128 * n_objects
    return 48 + 48 * n_objects


def prediction_record(ep: Episode, task: str, objects: dict, extents: dict | None,
                      parsed: bool, raw: str, frames, image_mode: str, style: str,
                      lora: str | None) -> dict:
    """One ``predictions.jsonl`` record; ``frames`` makes task N self-describing (t = first
    stored frame after the last input frame)."""
    rec = {
        "episode_id": ep.episode_id, "task": task, "objects": objects,
        "parsed": parsed, "raw": raw, "image_mode": image_mode,
        "style": style, "lora": lora, "frames": [int(f) for f in frames],
    }
    if extents is not None:
        rec["extents"] = extents
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=["A", "B"])
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--style", default="center", choices=["center", "qwen3d", "box3d", "chain3d", "ctx3d"])
    ap.add_argument("--image-mode", default="normal", choices=["normal", "none", "blank", "swap"])
    ap.add_argument("--frames", default=None, help="comma list of frame indices, e.g. 0,1,2")
    ap.add_argument("--views", default=None, help="comma list of view indices, e.g. 0")
    ap.add_argument("--no-pe", action="store_true", help="ignore pe.pt next to the adapter (B5)")
    ap.add_argument("--no-action", action="store_true", help="omit action text from prompt (infer from frames)")
    ap.add_argument("--oracle-context", action="store_true",
                    help="inject the simulator-provided action context into forward-prediction prompts")
    ap.add_argument("--direction-input", choices=("text", "nominal", "recorded"), default="text",
                    help="add a camera-frame direction vector to the forward-prediction prompt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="episodes per split")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames, views = parse_indices(args.frames), parse_indices(args.views)
    pairs = [(fr, vw) for fr in frames for vw in views]
    include_action = not args.no_action
    direction_input = args.direction_input if include_action else "text"
    model, processor = load_model(args.model, args.lora)
    merge = int(processor.image_processor.merge_size)
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
            work = []
            for d, ep in items:
                for task in args.tasks:
                    if task == IDP_TASK:  # legacy episodes lack the fixed 0.1 s pair
                        try:
                            idp_frame(ep)
                            ep.state_at(IDP_TIME)
                        except (KeyError, ValueError):
                            continue
                    work.append((d, ep, task))
            for i in tqdm(range(0, len(work), args.batch_size), desc=split):
                chunk = work[i:i + args.batch_size]
                msgs, xyz = [], []
                for d, ep, task in chunk:
                    # Task I reads its own fixed frame pair; everything else the arm's frames.
                    frames_ep = (0, idp_frame(ep)) if task == IDP_TASK else frames
                    pairs_ep = [(fr, vw) for fr in frames_ep for vw in views]
                    partner = partners.get(ep.episode_id)
                    images = input_images(d, ep, frames_ep, views, args.image_mode, partner)
                    msgs.append(build_messages(
                        prompt_text(ep, task, args.style, frames_ep, include_action=include_action,
                                    oracle_context=args.oracle_context,
                                    direction_input=direction_input),
                        images,
                    ))
                    if use_pe:
                        d_depth, ep_depth = partner if partner is not None else (d, ep)
                        grids = image_grid_rows(processor, images, grid_cache)
                        xyz.append(torch.cat([
                            frame_token_xyz(d_depth, ep_depth, frame=fr, view=vw,
                                            grid_thw_row=g, merge=merge)
                            for (fr, vw), g in zip(pairs_ep, grids)
                        ]))
                if pe is not None:
                    model._physwm_pe_xyz = torch.cat(xyz) if xyz else None
                max_new = token_budget(args.style, max(len(ep.objects) for _, ep, _ in chunk))
                answers = generate(model, processor, msgs, max_new)
                for (d, ep, task), ans in zip(chunk, answers):
                    frames_rec = (0, idp_frame(ep)) if task == IDP_TASK else frames
                    cur_objects, cur_extents = None, None
                    if task == IDP_TASK:
                        idp_ans, parsed = parse_idp(ans, ep)
                        rec = prediction_record(
                            ep, task, {}, None, parsed, ans, frames_rec,
                            image_mode=args.image_mode, style=args.style, lora=args.lora,
                        )
                        rec.pop("objects")
                        rec["idp"] = idp_ans
                    elif args.style == "ctx3d":
                        cur_objects, cur_extents, objects, extents, ctx, sup, _pc, parsed = \
                            parse_ctx3d(ans, ep)
                        rec = prediction_record(
                            ep, task, objects, extents, parsed, ans, frames_rec,
                            image_mode=args.image_mode, style=args.style, lora=args.lora,
                        )
                        rec["action_context"] = ctx
                        rec["support"] = sup
                    elif args.style == "chain3d":
                        cur_objects, cur_extents, objects, extents, _pc, parsed = parse_chain3d(ans, ep)
                        rec = prediction_record(
                            ep, task, objects, extents, parsed, ans, frames_rec,
                            image_mode=args.image_mode, style=args.style, lora=args.lora,
                        )
                    elif args.style == "box3d":
                        objects, extents, parsed = parse_box3d(ans, ep)
                        rec = prediction_record(
                            ep, task, objects, extents, parsed, ans, frames_rec,
                            image_mode=args.image_mode, style=args.style, lora=args.lora,
                        )
                    else:
                        objects, parsed = parse_answer(ans, ep)
                        extents = None
                        rec = prediction_record(
                            ep, task, objects, extents, parsed, ans, frames_rec,
                            image_mode=args.image_mode, style=args.style, lora=args.lora,
                        )
                    if cur_objects is not None:
                        rec["current_objects"] = cur_objects
                        rec["current_extents"] = cur_extents
                    if task in ("B", NEXT_FRAME_TASK) and (args.oracle_context or direction_input != "text"):
                        rec["direction_input"] = direction_input
                        rec["oracle_context"] = args.oracle_context
                    f.write(json.dumps(rec) + "\n")
                n_done += len(chunk)
    print(f"wrote {out_path} ({n_done} samples, {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
