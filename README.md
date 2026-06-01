# Sentiment Analysis — Full ML Pipeline

End-to-end 3-class sentiment analysis notebook comparing classical ML and transformer-based approaches, with a plug-and-play dataset configuration.

## Pipeline

| Stage | Details |
|---|---|
| Data | HuggingFace `datasets` or local CSV/Excel |
| EDA | Class distribution, text length, word clouds |
| Preprocessing | Lowercasing, URL/mention removal, stopwords, lemmatization |
| Classical ML | TF-IDF (unigrams + bigrams) · 5-fold CV across Logistic Regression, LinearSVC, Naive Bayes |
| Transformer | DistilBERT fine-tuned with HuggingFace `Trainer` |
| Evaluation | Classification report, confusion matrix, side-by-side accuracy comparison |
| Inference | Single `predict()` function runs both models on any input text |

## Setup

```bash
pip install -r requirements.txt
```

Then open `Sentiment_Analysis.ipynb` in Jupyter and run all cells top to bottom.

> **First run:** Cell 1 auto-detects and installs any missing packages. Restart the kernel once after installation.

## Using a Different Dataset

All settings are in the **Configuration** cell at the top of the notebook:

```python
# HuggingFace dataset
DATA_SOURCE    = 'huggingface'
DATASET_NAME   = 'tweet_eval'
DATASET_CONFIG = 'sentiment'

# — or — local CSV/Excel
DATA_SOURCE = 'csv'
CSV_PATH    = 'your_data.csv'

TEXT_COL  = 'text'
LABEL_COL = 'label'
LABEL_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}
```

Change those values and re-run — nothing else needs to be touched.

## Requirements

- Python 3.9+
- GPU optional (CPU works; transformer fine-tuning will be slower)

## Default Dataset

[Sp1786/multiclass-sentiment-analysis-dataset](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset) — 41,600 English samples across negative / neutral / positive classes (Apache 2.0).