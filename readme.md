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

### Dataset complet ou progressif

Commence avec 5 000 exemples pour valider le pipeline, puis augmente
progressivement :

```bash
# Premier test
python extract.py --mpd-path ./mpd/data --output-dir ./dataset --max-examples 5000

# Premier entraînement sérieux
python extract.py --mpd-path ./mpd/data --output-dir ./dataset-50k \
	--max-examples 50000 --validation-ratio 0.1 --seed 42

# MPD complet : 0 signifie sans limite
python extract.py --mpd-path ./mpd/data --output-dir ./dataset-full \
	--max-examples 0 --validation-ratio 0.1 --seed 42
```

Le MPD complet produit des fichiers volumineux et peut prendre du temps à
traiter. Ne l'ajoute pas à GitHub : conserve `mpd/`, `archive.zip` et
`dataset-full/` localement, ou copie uniquement les JSONL vers Google Drive.

Vérifie les fichiers générés :

```bash
wc -l dataset-50k/*.jsonl
ls -lh dataset-50k/
head -1 dataset-50k/train.jsonl | python3 -m json.tool
```

## Entraîner

L'entraînement LoRA se fait dans Colab avec un GPU, pas dans l'application
desktop Unsloth ni sur le CPU local. Ouvre le notebook depuis GitHub et choisis
`Runtime -> Change runtime type -> T4 GPU`.

Pour un petit dataset, téléverse directement `dataset/train.jsonl` et
`dataset/validation.jsonl` dans Colab. Pour un dataset de 50 000 exemples ou
le MPD complet, utilise Google Drive :

```python
from google.colab import drive
drive.mount("/content/drive")
```

Puis charge les fichiers avec leur chemin Drive :

```python
from datasets import load_dataset

dataset = load_dataset(
	"json",
	data_files="/content/drive/MyDrive/dataset-50k/train.jsonl",
	split="train",
)
validation_dataset = load_dataset(
	"json",
	data_files="/content/drive/MyDrive/dataset-50k/validation.jsonl",
	split="train",
)
```

Adapte `dataset-50k` en `dataset-full` si tu utilises le MPD complet.

Lance d'abord le notebook avec `max_steps=60` pour vérifier le modèle, le GPU
et le dataset. Si le test réussit, utilise `max_steps=-1` et
`num_train_epochs=1` pour l'entraînement complet. Une taille de 50 000 à
200 000 exemples est un bon compromis avant de passer au MPD complet.

Les scripts `mpd_to_jsonl.py`, `train_lora.py` et `generate.py` correspondent à
un ancien flux expérimental et ne sont pas le chemin de référence actuel.
