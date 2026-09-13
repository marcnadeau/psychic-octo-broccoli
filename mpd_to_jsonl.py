#!/usr/bin/env python3
"""
mpd_to_jsonl.py
---------------
Convert raw Spotify MPD JSON slices into a JSONL training file.

Usage:
    python mpd_to_jsonl.py \
        --in_dir  ./data/mpd_raw \
        --out     ./data/train.jsonl \
        --context_tracks 10 \
        --target_tracks  5  \
        --max_examples   50000
"""

import argparse
import json
import random
from pathlib import Path

PHI3_PROMPT = """\
<|user|>
Continue this playlist: {name}
Tracks so far:
{context}
<|end|>
<|assistant|>
{completion}
<|end|>"""


def format_track(i: int, track: dict) -> str:
    artist = track.get("artist_name", "Unknown Artist")
    name   = track.get("track_name",  "Unknown Track")
    return f"  {i}. {artist} – {name}"


def build_example(playlist: dict, context_n: int, target_n: int) -> str | None:
    tracks = playlist.get("tracks", [])
    if len(tracks) < context_n + target_n:
        return None
    ctx_tracks = tracks[:context_n]
    tgt_tracks = tracks[context_n:context_n + target_n]
    pl_name    = playlist.get("name", "Unnamed Playlist")
    context    = "\n".join(format_track(i + 1, t) for i, t in enumerate(ctx_tracks))
    completion = "\n".join(
        format_track(context_n + i + 1, t) for i, t in enumerate(tgt_tracks)
    )
    return PHI3_PROMPT.format(name=pl_name, context=context, completion=completion)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir",         type=Path,  default=Path("./data/mpd_raw"))
    p.add_argument("--out",            type=Path,  default=Path("./data/train.jsonl"))
    p.add_argument("--val_out",        type=Path,  default=Path("./data/val.jsonl"))
    p.add_argument("--context_tracks", type=int,   default=10)
    p.add_argument("--target_tracks",  type=int,   default=5)
    p.add_argument("--min_tracks",     type=int,   default=15)
    p.add_argument("--max_examples",   type=int,   default=0)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--val_split",      type=float, default=0.02)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    slices = sorted(args.in_dir.glob("mpd.slice.*.json"))
    if not slices:
        raise FileNotFoundError(
            f"No mpd.slice.*.json files found in {args.in_dir}. "
            "Run mpd_download.py first."
        )
    print(f"[mpd_to_jsonl] Found {len(slices)} slice(s)")

    examples: list[str] = []
    for slice_path in slices:
        with open(slice_path, encoding="utf-8") as fh:
            data = json.load(fh)
        for playlist in data.get("playlists", []):
            if len(playlist.get("tracks", [])) < args.min_tracks:
                continue
            ex = build_example(playlist, args.context_tracks, args.target_tracks)
            if ex is not None:
                examples.append(ex)
        if args.max_examples > 0 and len(examples) >= args.max_examples:
            examples = examples[:args.max_examples]
            print(f"[mpd_to_jsonl] Hit --max_examples cap: {args.max_examples}")
            break

    print(f"[mpd_to_jsonl] Total examples: {len(examples)}")
    random.shuffle(examples)

    val_n    = max(1, int(len(examples) * args.val_split))
    val_ex   = examples[:val_n]
    train_ex = examples[val_n:]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for ex in train_ex:
            fh.write(json.dumps({"text": ex}, ensure_ascii=False) + "\n")
    print(f"[mpd_to_jsonl] Train → {args.out}  ({len(train_ex)} examples)")

    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.val_out, "w", encoding="utf-8") as fh:
        for ex in val_ex:
            fh.write(json.dumps({"text": ex}, ensure_ascii=False) + "\n")
    print(f"[mpd_to_jsonl] Val   → {args.val_out}  ({len(val_ex)} examples)")

    print("[mpd_to_jsonl] Done.")


if __name__ == "__main__":
    main()
