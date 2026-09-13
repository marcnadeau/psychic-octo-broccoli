# UnsLoTH Music Pipeline

Fine-tune **Phi-3-mini-4k-instruct** on the Spotify Million Playlist Dataset
using LoRA + 4-bit quantisation — runs entirely on CPU.

```
unsloth_music/
├── mpd_download.py   # Step 1 – Download & extract MPD
├── mpd_to_jsonl.py   # Step 2 – Convert JSON → JSONL
├── train_lora.py     # Step 3 – Fine-tune Phi-3 Mini with LoRA
├── generate.py       # Step 4 – Generate playlist continuations
└── README.md
```

## Install

```bash
pip install unsloth "unsloth[colab-new]"
pip install bitsandbytes trl transformers accelerate datasets peft aicrowd-cli
```

## Step 1 — Download MPD

```bash
export AICROWD_API_TOKEN=your_token_here
python mpd_download.py --out_dir ./data/mpd_raw

# Already have the zip?
python mpd_download.py --local_zip /path/to/mpd.zip --out_dir ./data/mpd_raw

# Quick test: only 5 slices
python mpd_download.py --local_zip /path/to/mpd.zip --out_dir ./data/mpd_raw --max_slices 5
```

## Step 2 — Convert to JSONL

```bash
python mpd_to_jsonl.py \
    --in_dir ./data/mpd_raw \
    --context_tracks 10 \
    --target_tracks  5  \
    --max_examples   50000
```

Each line: `{"text": "<phi3-chat prompt+completion>"}`.

## Step 3 — Train

```bash
python train_lora.py \
    --train_file ./data/train.jsonl \
    --val_file   ./data/val.jsonl
```

| Param | Value |
|---|---|
| Model | Phi-3-mini-4k-instruct |
| LoRA r | 8 |
| LoRA alpha | 16 |
| Targets | q_proj, v_proj |
| Seq len | 1024 |
| Quant | 4-bit NF4 |
| Device | CPU |

Add `--no_4bit` for faster dry-runs. Add `--merge_adapter` to bake weights in.

## Step 4 — Generate

```bash
# Interactive
python generate.py --adapter ./outputs/phi3-music-lora/lora_adapter

# Single playlist
python generate.py \
    --adapter ./outputs/phi3-music-lora/lora_adapter \
    --name "Late Night Drive" \
    --tracks "Radiohead – No Surprises" "Portishead – Glory Box" \
    --n_generate 5 --out_json ./outputs/result.json

# Batch (JSONL: {"name":…,"tracks":[…],"n_generate":5})
python generate.py --adapter ./outputs/phi3-music-lora/lora_adapter \
    --batch_file requests.jsonl --out_json ./outputs/batch.json
```

## End-to-End Quick Test

```bash
python mpd_download.py --local_zip ~/Downloads/mpd.zip \
    --out_dir ./data/mpd_raw --max_slices 5
python mpd_to_jsonl.py --in_dir ./data/mpd_raw --max_examples 2000
python train_lora.py --no_4bit --max_train_steps 100
python generate.py --adapter ./outputs/phi3-music-lora/lora_adapter \
    --no_4bit --name "Sunday Morning Coffee" \
    --tracks "Norah Jones – Don't Know Why" "John Mayer – Gravity"
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `AICROWD_API_TOKEN not set` | `export AICROWD_API_TOKEN=<token>` |
| No slice files found | Run `mpd_download.py` first |
| Adapter not found | Run `train_lora.py` first |
| OOM on CPU | `--batch_size 1 --no_4bit` |
| Slow training | Expected on CPU; use `--max_train_steps 100` to test |
