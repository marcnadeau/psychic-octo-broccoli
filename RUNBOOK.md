# Runbook

## 1. Prepare the local project

The large MPD archive and raw slices stay on the local machine. They are not
committed to Git.

```bash
cd /home/marc/code/psychic-octo-broccoli
python3 -m venv .venv
source .venv/bin/activate
python -m py_compile extract.py
```

If the archive is named `archive.zip` and contains `data/*.json`, extract only
those JSON files:

```bash
mkdir -p mpd
unzip -q archive.zip "data/*.json" -d mpd
```

The resulting files should be under `mpd/data/`.

## 2. Generate the training data

Generate a small dataset first to verify the pipeline:

```bash
.venv/bin/python extract.py \
  --mpd-path ./mpd/data \
  --output-dir ./dataset \
  --max-examples 5000 \
  --validation-ratio 0.1 \
  --seed 42
```

Expected files:

```text
dataset/train.jsonl
dataset/validation.jsonl
```

Validate the output:

```bash
wc -l dataset/*.jsonl
head -1 dataset/train.jsonl | python3 -m json.tool
```

Each record must contain exactly these source columns:

```text
instruction, input, output
```

## 3. Push only the useful files

The Colab notebook needs the two generated JSONL files. Upload or commit:

```text
dataset/train.jsonl
dataset/validation.jsonl
nb/Llama3_(8B)-Ollama.ipynb
```

Never commit:

```text
archive.zip
mpd/
.venv/
kaggle.json
outputs/
```

Check before pushing:

```bash
git status --short
git add .gitignore readme.md RUNBOOK.md extract.py dataset nb
# Inspect the staged files before committing.
git diff --cached --stat
git commit -m "Document playlist LoRA training pipeline"
git push
```

## 4. Open the notebook in Colab

Open the notebook's **Open in Colab** link from GitHub. Select a GPU runtime:

```text
Runtime -> Change runtime type -> T4 GPU
```

Upload the `dataset` folder into the notebook root. Confirm the files exist:

```python
!find dataset -maxdepth 1 -type f -name '*.jsonl' -print
```

## 5. Load the dataset

Replace the example Hugging Face dataset cell with:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="dataset/train.jsonl",
    split="train",
)
validation_dataset = load_dataset(
    "json",
    data_files="dataset/validation.jsonl",
    split="train",
)

print(dataset.column_names)
print(len(dataset), len(validation_dataset))
```

Expected output:

```text
['instruction', 'input', 'output']
```

## 6. Prepare prompts

Keep the notebook's `to_sharegpt`, `standardize_sharegpt`, and
`apply_chat_template` cells. Run them once, in that order, after loading the
model and tokenizer. Apply the same preparation to validation if evaluation is
enabled:

```python
from unsloth import to_sharegpt, standardize_sharegpt, apply_chat_template

for name in ("dataset", "validation_dataset"):
    prepared = to_sharegpt(
        globals()[name],
        merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
        output_column_name="output",
        conversation_extension=1,
    )
    prepared = standardize_sharegpt(prepared)
    prepared = apply_chat_template(
        prepared,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
    globals()[name] = prepared

print(dataset.column_names)
print(dataset[0]["text"][:1000])
```

The dataset must contain a `text` column before creating the trainer.

## 7. Train a short smoke test

Keep only one `FastLanguageModel.get_peft_model(...)` cell; running it twice
can apply LoRA adapters twice. In the trainer cell, start with:

```python
max_steps = 10
```

Then run:

```python
trainer_stats = trainer.train()
```

If that succeeds, use a real run:

```python
max_steps = -1
num_train_epochs = 1
```

For a 12 GB GPU, start conservatively with batch size 1, gradient accumulation
4, and sequence length 1024. For a free Colab T4, 5,000 examples and one epoch
are a reasonable first run.

## 8. Save the adapter

```python
model.save_pretrained("playlist_lora")
tokenizer.save_pretrained("playlist_lora")
```

Download `playlist_lora/` from Colab. The desktop app can be used later for
local inference; it is not the training environment for this workflow.

## Troubleshooting

- `FileNotFoundError`: run `!find /content -name '*.jsonl'` and correct the
  `data_files` path.
- Missing `text`: run the prompt preparation cells before `SFTTrainer`.
- CUDA or GPU errors: select a T4 GPU runtime and restart the runtime.
- Duplicate LoRA adapter errors: restart the runtime and run the LoRA cell only
  once.
- Git rejects a large file: remove it from staging and verify `.gitignore`.
