# E-Commerce Aspect-Based Sentiment Analysis

An end-to-end **Aspect-Based Sentiment Analysis (ABSA)** system for e-commerce product reviews. It extracts aspect terms from review text, classifies their sentiment polarity, and surfaces product strengths and weaknesses through a clean web UI.

---

## What it does

Given a product review like:
> *"The battery life is great but the keyboard feels cheap."*

The system returns:

| Aspect | Polarity | Confidence |
|---|---|---|
| battery life | Positive | 94.2% |
| keyboard | Negative | 87.6% |

Along with a summarized view of **Strengths** and **Weaknesses** across reviews.

---

## Architecture

```
Review Text
    → Preprocessor (normalize)
    → BERT Tokenizer
    → BERT Encoder (bert-base-uncased)
    → Aspect Extractor (BIO tagging head)
    → Sentiment Classifier (per-aspect pooling head)
    → Pair Builder
    → Opinion Summarizer
    → JSON output
```

- **Backend**: FastAPI + PyTorch
- **Frontend**: React 19 + Vite
- **Model**: Fine-tuned `bert-base-uncased` on SemEval laptop reviews dataset

---

## Project Structure

```
├── src/absa/           # Core pipeline modules
│   ├── pipeline.py     # End-to-end inference pipeline
│   ├── aspect_extractor.py
│   ├── sentiment_classifier.py
│   ├── encoder.py      # BERT encoder wrapper
│   ├── tokenizer.py
│   ├── pair_builder.py
│   ├── opinion_summarizer.py
│   ├── training_orchestrator.py
│   └── config.py
├── frontend/           # React UI
├── api.py              # FastAPI server
├── config.yaml         # Training & inference config
├── data/raw/           # SemEval XML datasets
├── models/             # Model checkpoints (not tracked in git)
└── tests/              # Unit, property, and benchmark tests
```

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend

```bash
pip install -r requirements.txt
pip install -e .
```

### Frontend

```bash
cd frontend
npm install
```

---

## Model Weights

Model weights (`models/best_model.pt`) are not tracked in this repo due to file size.

To get the weights:
- **Train from scratch** using the instructions below, or
- Download from [releases / shared link — add yours here]

---

## Training

```bash
python -m absa.training_orchestrator
```

Key config options in `config.yaml`:

```yaml
bert_checkpoint: "bert-base-uncased"
learning_rate: 2.0e-5
batch_size: 32
num_epochs: 10
train_split: 0.8
val_split: 0.1
test_split: 0.1
confidence_threshold: 0.5
```

---

## Running

**Start the API server:**

```bash
uvicorn api:app --reload
```

**Start the frontend:**

```bash
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

**CLI inference:**

```bash
python -m absa.pipeline --review "The battery life is great but the keyboard feels cheap"
```

---

## API

### `POST /analyze`

```json
{
  "review": "The screen is stunning but the fan noise is annoying."
}
```

Response:

```json
{
  "review": "...",
  "aspects": [
    { "term": "screen", "polarity": "Positive", "confidence": 0.95, "low_confidence": false },
    { "term": "fan noise", "polarity": "Negative", "confidence": 0.88, "low_confidence": false }
  ],
  "summary": {
    "strengths": [{ "aspect": "screen", "count": 1 }],
    "weaknesses": [{ "aspect": "fan noise", "count": 1 }]
  }
}
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## Tests

```bash
pytest tests/
```

---

## Dataset

Trained on the [SemEval 2014 Task 4](https://alt.qcri.org/semeval2014/task4/) laptop reviews dataset.
