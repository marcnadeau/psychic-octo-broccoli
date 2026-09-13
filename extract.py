import json
import os
import random
import argparse
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
MPD_PATH = "./mpd"            # dossier contenant les fichiers du MPD
OUTPUT_DIR = "./dataset"      # fichiers train.jsonl et validation.jsonl
SEED_MIN = 3
SEED_MAX = 5
EXPECTED_MIN = 3
EXPECTED_MAX = 5
MAX_DATASET_SIZE = 5000       # nombre total d'exemples à générer
VALIDATION_RATIO = 0.1
DEFAULT_SEED = 42
# ---------------------------------------------------------

def load_mpd_files(path):
    """Retourne les fichiers MPD dans un ordre déterministe."""
    return sorted(Path(path).glob("*.json"))

def song_name(track):
    """Gère les pistes normales et les éventuels champs manquants."""
    artist = track.get("artist_name", "Artiste inconnu")
    title = track.get("track_name", "Titre inconnu")
    return f"{artist} – {title}"


def extract_playlist_example(playlist, rng):
    tracks = playlist.get("tracks", [])
    if len(tracks) < SEED_MIN + EXPECTED_MIN:
        return None

    # Les premières pistes servent de contexte; les suivantes sont la cible.
    seed_len = rng.randint(SEED_MIN, min(SEED_MAX, len(tracks) - EXPECTED_MIN))
    expected_max = min(EXPECTED_MAX, len(tracks) - seed_len)
    expected_len = rng.randint(EXPECTED_MIN, expected_max)
    seed = tracks[:seed_len]
    expected = tracks[seed_len:seed_len + expected_len]

    # Format Alpaca attendu par de nombreux notebooks Unsloth:
    # instruction / input / output.
    title = playlist.get("name", "sans titre")
    return {
        "instruction": (
            "Tu es un assistant expert en continuation de playlists. "
            "Propose les chansons suivantes et réponds uniquement avec une liste "
            "JSON au format Artiste – Titre."
        ),
        "input": (
            f'Playlist intitulée « {title} ». Chansons déjà présentes :\n'
            + "\n".join(f"{i + 1}. {song_name(t)}" for i, t in enumerate(seed))
            + f"\nPropose les {expected_len} chansons suivantes."
        ),
        "output": json.dumps([song_name(t) for t in expected], ensure_ascii=False),
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Convertit le MPD en JSONL pour Unsloth.")
    parser.add_argument("--mpd-path", default=MPD_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-examples", type=int, default=MAX_DATASET_SIZE)
    parser.add_argument("--validation-ratio", type=float, default=VALIDATION_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.validation_ratio < 1:
        raise ValueError("--validation-ratio doit être compris entre 0 et 1.")

    files = load_mpd_files(args.mpd_path)
    if not files:
        raise FileNotFoundError(f"Aucun fichier JSON trouvé dans {args.mpd_path}")
    print(f"Found {len(files)} MPD files.")

    rng = random.Random(args.seed)
    train_path = Path(args.output_dir) / "train.jsonl"
    validation_path = Path(args.output_dir) / "validation.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    count = train_count = validation_count = 0
    with train_path.open("w", encoding="utf-8") as train_out, \
            validation_path.open("w", encoding="utf-8") as validation_out:
        for file in files:
            print(f"Processing {file}...")
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            playlists = data.get("playlists", [])
            for playlist in playlists:
                example = extract_playlist_example(playlist, rng)
                if example:
                    # Un même playlist ne peut pas se retrouver dans train et validation.
                    output = validation_out if rng.random() < args.validation_ratio else train_out
                    output.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1
                    if output is validation_out:
                        validation_count += 1
                    else:
                        train_count += 1

                if count >= args.max_examples:
                    break
            if count >= args.max_examples:
                break

    print(f"Dataset generated: {count} examples")
    print(f"Train: {train_path} ({train_count})")
    print(f"Validation: {validation_path} ({validation_count})")

if __name__ == "__main__":
    main()
