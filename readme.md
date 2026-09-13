# Playlist LoRA

Pipeline pour préparer le Spotify Million Playlist Dataset et fine-tuner Phi-3
avec LoRA dans le notebook Unsloth/Colab.

Le chemin recommandé est documenté dans [RUNBOOK.md](RUNBOOK.md). Le dépôt
contient déjà le notebook [nb/Llama3_(8B)-Ollama.ipynb](nb/Llama3_(8B)-Ollama.ipynb).

## Structure

```text
extract.py                         # MPD JSON -> Alpaca JSONL
dataset/train.jsonl                # données d'entraînement générées
dataset/validation.jsonl           # données de validation générées
nb/Llama3_(8B)-Ollama.ipynb        # entraînement dans Colab
```

Les données brutes (`mpd/`), l'archive MPD et les sorties de modèle restent
locales et sont ignorées par Git.

## Générer les données

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m py_compile extract.py
python extract.py --mpd-path ./mpd/data --output-dir ./dataset --max-examples 5000
```

Le dataset produit contient les colonnes `instruction`, `input` et `output`.
Il peut être chargé avec `datasets.load_dataset("json", ...)`.

## Entraîner

L'entraînement LoRA se fait dans Colab avec un GPU, pas dans l'application
desktop Unsloth ni sur le CPU local. Ouvre le notebook depuis GitHub, ajoute
`dataset/train.jsonl` et `dataset/validation.jsonl`, puis suis le runbook.

Les scripts `mpd_to_jsonl.py`, `train_lora.py` et `generate.py` correspondent à
un ancien flux expérimental et ne sont pas le chemin de référence actuel.
