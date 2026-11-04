# SML NLP Project

A staged text-classification study on the AG News dataset, comparing sparse
text features with contextual transformer embeddings and classical machine
learning models.

## Requirements

Python 3.10 or newer is recommended. Install the project dependencies in a
virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

The launcher runs each stage separately:

```bash
python run.py part1
python run.py part2 --encoder microsoft/deberta-v3-base
python run.py compare-encoders
python run.py part3 --encoder microsoft/deberta-v3-base
```

- `part1` evaluates sparse features and classical models.
- `part2` downloads the AG News dataset and selected encoder as needed, then
  extracts and caches contextual embeddings.
- `compare-encoders` compares encoders for which embeddings have been cached.
- `part3` trains and evaluates classical models on a selected encoder's
  cached embeddings.

The embedding stages can download large models and benefit from a GPU. The
encoder argument supports `distilbert-base-uncased`, `bert-base-uncased`,
`roberta-base`, and `microsoft/deberta-v3-base`.

## Outputs

The current results include a report (`report.pdf`), figures, and result
tables. Running the pipeline can create dataset, embedding-cache, and model
artifacts under `results/`; these generated files are excluded from Git.
Configuration such as dataset sample size and cross-validation folds is in
`code/config.py`.

## Reproducibility

The project uses the AG News dataset, an 8,000-example sample, a fixed random
seed, and three-fold cross-validation by default. See `code/config.py` and
the included report for the experiment setup and results. Model downloads
and hardware can affect runtime.
