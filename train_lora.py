#!/usr/bin/env python3
"""
train_lora.py
-------------
Fine-tune Phi-3-mini-4k-instruct with LoRA via UnsLoTH.

  Model  : microsoft/Phi-3-mini-4k-instruct
  LoRA   : r=8, alpha=16, q_proj+v_proj, dropout=0.05
  Quant  : 4-bit NF4 (bitsandbytes)
  SeqLen : 1024
  Device : CPU

Usage:
    python train_lora.py \
        --train_file ./data/train.jsonl \
        --val_file   ./data/val.jsonl   \
        --output_dir ./outputs/phi3-music-lora
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--train_file", type=Path,  default=Path("./data/train.jsonl"))
    p.add_argument("--val_file",   type=Path,  default=Path("./data/val.jsonl"))
    # Model
    p.add_argument("--model_name", default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--seq_len",    type=int,   default=1024)
    p.add_argument("--no_4bit",    action="store_true")
    # LoRA
    p.add_argument("--lora_r",       type=int,   default=8)
    p.add_argument("--lora_alpha",   type=int,   default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", nargs="+",  default=["q_proj", "v_proj"])
    # Training
    p.add_argument("--output_dir",      type=Path,  default=Path("./outputs/phi3-music-lora"))
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--batch_size",      type=int,   default=2)
    p.add_argument("--grad_accum",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=2e-4)
    p.add_argument("--warmup_steps",    type=int,   default=50)
    p.add_argument("--save_steps",      type=int,   default=200)
    p.add_argument("--logging_steps",   type=int,   default=20)
    p.add_argument("--max_train_steps", type=int,   default=0)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--merge_adapter",   action="store_true")
    return p.parse_args()


def load_jsonl(path: Path):
    from datasets import load_dataset
    return load_dataset("json", data_files=str(path), split="train")


def main() -> None:
    args = parse_args()

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    print(f"[train_lora] Model: {args.model_name} | 4-bit: {not args.no_4bit} | device: CPU")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.model_name,
        max_seq_length = args.seq_len,
        dtype          = None,
        load_in_4bit   = not args.no_4bit,
        device_map     = "cpu",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                          = args.lora_r,
        lora_alpha                 = args.lora_alpha,
        lora_dropout               = args.lora_dropout,
        target_modules             = args.lora_targets,
        bias                       = "none",
        use_gradient_checkpointing = True,
        random_state               = args.seed,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[train_lora] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    if not args.train_file.exists():
        raise FileNotFoundError(f"Not found: {args.train_file} — run mpd_to_jsonl.py first.")

    train_ds = load_jsonl(args.train_file)
    eval_ds  = load_jsonl(args.val_file) if args.val_file.exists() else None
    print(f"[train_lora] Train: {len(train_ds):,}" + (f" | Val: {len(eval_ds):,}" if eval_ds else ""))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = SFTConfig(
        output_dir                  = str(args.output_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        warmup_steps                = args.warmup_steps,
        save_steps                  = args.save_steps,
        logging_steps               = args.logging_steps,
        max_steps                   = args.max_train_steps if args.max_train_steps > 0 else -1,
        seed                        = args.seed,
        fp16                        = False,
        bf16                        = False,
        optim                       = "adamw_torch",
        dataset_text_field          = "text",
        max_seq_length              = args.seq_len,
        packing                     = True,
        report_to                   = "none",
        save_total_limit            = 3,
        load_best_model_at_end      = eval_ds is not None,
        evaluation_strategy         = "steps" if eval_ds else "no",
        eval_steps                  = args.save_steps if eval_ds else None,
    )

    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        args          = cfg,
    )

    print("[train_lora] Training …")
    trainer.train()
    print("[train_lora] Training complete.")

    adapter_dir = args.output_dir / "lora_adapter"
    if args.merge_adapter:
        merged_dir = args.output_dir / "merged_model"
        print(f"[train_lora] Merging → {merged_dir}")
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    else:
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"[train_lora] Adapter saved → {adapter_dir}")

    print("[train_lora] Done.")


if __name__ == "__main__":
    main()
