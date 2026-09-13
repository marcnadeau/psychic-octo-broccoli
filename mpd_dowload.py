#!/usr/bin/env python3
"""
mpd_download.py
---------------
Download the Spotify Million Playlist Dataset (MPD) challenge files and
store them locally as raw JSON slices ready for mpd_to_jsonl.py.

Usage:
    python mpd_download.py --out_dir ./data/mpd_raw

The MPD is distributed via the AIcrowd challenge.  You need:
  1. A free AIcrowd account  →  https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
  2. To accept the dataset Terms of Use on that page.
  3. Your AIcrowd API token  (Profile → Settings → API Key).

Set the token via environment variable or --token flag:
    export AICROWD_API_TOKEN=your_token_here
    python mpd_download.py --out_dir ./data/mpd_raw

Alternatively, if you already have the MPD .zip downloaded locally, pass it:
    python mpd_download.py --local_zip /path/to/spotify_million_playlist_dataset.zip --out_dir ./data/mpd_raw
"""

import argparse
import os
import sys
import zipfile
import json
from pathlib import Path

try:
    from aicrowd.api import AIcrowdAPI
    AICROWD_AVAILABLE = True
except ImportError:
    AICROWD_AVAILABLE = False

CHALLENGE_ID  = "spotify-million-playlist-dataset-challenge"
DATASET_SLUG  = "spotify_million_playlist_dataset"


def _token_from_env() -> str:
    token = os.environ.get("AICROWD_API_TOKEN", "").strip()
    if not token:
        raise EnvironmentError(
            "AICROWD_API_TOKEN is not set. "
            "Export it or pass --token <your_token>."
        )
    return token


def download_via_aicrowd(out_dir: Path, token: str) -> Path:
    if not AICROWD_AVAILABLE:
        raise ImportError("Run:  pip install aicrowd-cli")
    api = AIcrowdAPI(api_key=token)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[mpd_download] Fetching dataset files from AIcrowd …")
    files = api.dataset_list_files(challenge_id=CHALLENGE_ID)
    for f in files:
        if DATASET_SLUG in f.get("name", "").lower():
            url  = f["url"]
            name = f["name"]
            dest = out_dir / name
            print(f"[mpd_download] Downloading {name} → {dest}")
            api.download_file(url, str(dest))
            return dest
    raise RuntimeError("Could not locate the MPD zip on AIcrowd.")


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[mpd_download] Extracting {zip_path} → {out_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".json")]
        total   = len(members)
        for i, member in enumerate(members, 1):
            fname = Path(member).name
            dest  = out_dir / fname
            if dest.exists():
                print(f"  [{i}/{total}] skip (exists): {fname}")
                continue
            print(f"  [{i}/{total}] extract: {fname}")
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
    print(f"[mpd_download] Extraction complete. {total} JSON slices in {out_dir}")


def validate_slices(out_dir: Path, max_check: int = 3) -> None:
    slices = sorted(out_dir.glob("mpd.slice.*.json"))[:max_check]
    if not slices:
        print("[mpd_download] WARNING: no mpd.slice.*.json files found.")
        return
    for s in slices:
        with open(s) as fh:
            data = json.load(fh)
        n = len(data.get("playlists", []))
        print(f"[mpd_download] {s.name}: {n} playlists OK")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download / extract the Spotify MPD dataset.")
    p.add_argument("--out_dir",    type=Path, default=Path("./data/mpd_raw"))
    p.add_argument("--token",      default="")
    p.add_argument("--local_zip",  type=Path, default=None)
    p.add_argument("--keep_zip",   action="store_true")
    p.add_argument("--max_slices", type=int,  default=0)
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    out_dir  = args.out_dir
    zip_path = args.local_zip

    if zip_path is None:
        token    = args.token.strip() or _token_from_env()
        zip_path = download_via_aicrowd(out_dir.parent / "downloads", token)

    if not zip_path.exists():
        sys.exit(f"[mpd_download] ERROR: zip not found at {zip_path}")

    extract_zip(zip_path, out_dir)

    if args.max_slices > 0:
        all_slices = sorted(out_dir.glob("mpd.slice.*.json"))
        for s in all_slices[args.max_slices:]:
            s.unlink()
        print(f"[mpd_download] Trimmed to {args.max_slices} slices.")

    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()
        print(f"[mpd_download] Removed zip: {zip_path}")

    validate_slices(out_dir)
    print("[mpd_download] Done.")


if __name__ == "__main__":
    main()
