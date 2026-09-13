#!/usr/bin/env python3
"""
generate.py
-----------
Generate playlist continuations with the fine-tuned Phi-3 Mini LoRA model.

Interactive:
    python generate.py --adapter ./outputs/phi3-music-lora/lora_adapter

Single playlist:
    python generate.py \
        --adapter ./outputs/phi3-music-lora/lora_adapter \
        --name    "Late Night Drive" \
        --tracks  "Radiohead – No Surprises" "Portishead – Glory Box" \
        --n_generate 5 --out_json ./outputs/continuation.json

Batch (JSONL file, one {"name":…,"tracks":[…],"n_generate":5} per line):
    python generate.py --adapter ./outputs/phi3-music-lora/lora_adapter \
        --batch_file requests.jsonl --out_json ./outputs/batch_results.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PHI3_PROMPT_TMPL = """\
<|user|>
Continue this playlist: {name}
Tracks so far:
{context}
<|end|>
<|assistant|>
"""


def build_prompt(name: str, tracks: list[str]) -> str:
    context = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(tracks))
    return PHI3_PROMPT_TMPL.format(name=name, context=context)


def parse_generated_tracks(text: str) -> list[str]:
    tracks = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^\d+\.\s*(.+)", line)
        if m:
            tracks.append(m.group(1).strip())
        elif " – " in line and line:
            tracks.append(line)
    return tracks


def load_model(adapter_path: Path, base_model: str, use_4bit: bool):
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = str(adapter_path),
            max_seq_length = 1024,
            dtype          = None,
            load_in_4bit   = use_4bit,
            device_map     = "cpu",
        )
        FastLanguageModel.for_inference(model)
        print("[generate] Loaded via UnsLoTH.")
        return model, tokenizer
    except ImportError:
        pass

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    print("[generate] Falling back to standard PEFT + transformers.")
    bnb_cfg = None
    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float32,
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_cfg, device_map="cpu", trust_remote_code=True,
    )
    if (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        model = base
    model.eval()
    return model, tokenizer


def generate_continuation(model, tokenizer, name, tracks, n_generate,
                           max_new_tokens, temperature, top_p, top_k) -> list[str]:
    import torch
    prompt    = build_prompt(name, tracks)
    inputs    = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_p          = top_p,
            top_k          = top_k,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
            eos_token_id   = tokenizer.convert_tokens_to_ids("<|end|>"),
        )
    new_ids  = out[0][inputs["input_ids"].shape[-1]:]
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return parse_generated_tracks(raw_text)[:n_generate]


def interactive_session(model, tokenizer, args):
    print("\n[generate] ── Interactive Playlist Continuation ── (type 'quit' to exit)\n")
    while True:
        name = (args.name or input("Playlist name: ")).strip()
        args.name = None
        if name.lower() == "quit":
            break
        if args.tracks:
            tracks = list(args.tracks)
            args.tracks = None
        else:
            print("Seed tracks (blank line to finish):")
            tracks = []
            while True:
                line = input(f"  Track {len(tracks)+1}: ").strip()
                if not line:
                    break
                tracks.append(line)
        if not tracks:
            continue
        generated = generate_continuation(
            model, tokenizer, name, tracks, args.n_generate,
            args.max_new_tokens, args.temperature, args.top_p, args.top_k,
        )
        print(f'\n  ── Continuation for "{name}" ──')
        for i, t in enumerate(generated, start=len(tracks)+1):
            print(f"  {i}. {t}")
        print()
        if args.out_json:
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump({"playlist_name": name, "seed_tracks": tracks,
                           "continuation": generated}, fh, indent=2)
            print(f"  Saved → {out_path}\n")
            args.out_json = None


def batch_mode(model, tokenizer, args):
    batch_path = Path(args.batch_file)
    with open(batch_path) as fh:
        requests = [json.loads(l) for l in fh if l.strip()]
    print(f"[generate] Batch: {len(requests)} request(s)")
    records = []
    for i, req in enumerate(requests, 1):
        name   = req.get("name", "Unnamed")
        tracks = req.get("tracks", [])
        n      = req.get("n_generate", args.n_generate)
        print(f"  [{i}/{len(requests)}] {name}")
        generated = generate_continuation(
            model, tokenizer, name, tracks, n,
            args.max_new_tokens, args.temperature, args.top_p, args.top_k,
        )
        records.append({"playlist_name": name, "seed_tracks": tracks, "continuation": generated})
        for j, t in enumerate(generated, start=len(tracks)+1):
            print(f"    {j}. {t}")
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(records, fh, indent=2)
        print(f"[generate] Results saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",        type=Path,  default=Path("./outputs/phi3-music-lora/lora_adapter"))
    p.add_argument("--base_model",     default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--name",           default="")
    p.add_argument("--tracks",         nargs="*")
    p.add_argument("--n_generate",     type=int,   default=5)
    p.add_argument("--max_new_tokens", type=int,   default=256)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--top_p",          type=float, default=0.9)
    p.add_argument("--top_k",          type=int,   default=50)
    p.add_argument("--no_4bit",        action="store_true")
    p.add_argument("--out_json",       default="")
    p.add_argument("--batch_file",     default="")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.adapter.exists():
        sys.exit(f"[generate] Adapter not found: {args.adapter} — run train_lora.py first.")
    print(f"[generate] Loading {args.adapter} …")
    model, tokenizer = load_model(args.adapter, args.base_model, not args.no_4bit)
    if args.batch_file:
        batch_mode(model, tokenizer, args)
    else:
        interactive_session(model, tokenizer, args)
    print("[generate] Done.")


if __name__ == "__main__":
    main()
