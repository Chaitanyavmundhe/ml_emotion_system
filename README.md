# ML Emotion System

An intelligent emotion understanding and decision-making system that takes
user reflections and contextual signals to predict emotional states,
recommend meaningful actions, and guide users toward better mental states.

---

## Project Structure
```
ml_emotion_system/
├── api/
│   └── app.py                  # FastAPI server
├── data/
│   ├── raw/                    # Original dataset (not tracked in git)
│   └── processed/              # Cleaned and transformed data
├── models/                     # Saved trained models (not tracked in git)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_ablation_study.ipynb # Ablation study
│   ├── 03_error_analysis.ipynb # Error analysis
│   └── 04_improvement.py       # Accuracy improvement experiments
├── outputs/
│   ├── predictions.csv         # Final predictions
│   └── ablation_study.png      # Ablation study chart
├── src/
│   ├── preprocess.py           # Data cleaning and encoding
│   ├── features.py             # TF-IDF + metadata feature engineering
│   ├── embeddings.py           # Sentence transformer embeddings + ensemble
│   ├── train.py                # Model training (XGBoost + Random Forest)
│   ├── predict.py              # TF-IDF prediction pipeline
│   ├── recommend.py            # Decision engine (what + when)
│   └── uncertainty.py          # Confidence scoring and uncertain flags
├── ui/
│   └── index.html              # React UI demo
├── ERROR_ANALYSIS.md           # 10 failure case analysis
├── EDGE_PLAN.md                # Mobile and offline deployment plan
├── main.py                     # End-to-end ensemble pipeline entry point
└── requirements.txt            # Python dependencies
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ml_emotion_system.git
cd ml_emotion_system
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add dataset
Place `train.csv` and `test.csv` into `data/raw/`.

---

## How to Run

### Run full pipeline at once (recommended)
```bash
python main.py
```

### Step by step

#### Step 1 — Preprocess
```bash
python src/preprocess.py
```

#### Step 2 — Train base models
```bash
python src/train.py
```

#### Step 3 — Generate ensemble predictions
```bash
python src/embeddings.py
```

### Run API server
```bash
uvicorn api.app:app --reload
```

API docs available at: `http://localhost:8000/docs`

### Open UI
Open `ui/index.html` in your browser while the API server is running.

Output saved to `outputs/predictions.csv`.

---

## Approach

### Problem Understanding
Users write short reflections (avg 11 words) after immersive sessions.
These reflections are messy, vague, and sometimes contradictory.
The system must understand emotional state and guide users — not just classify.

### Key Finding from EDA
The same text ("okay session", "felt heavy") maps to completely different
emotional states depending on context. This means:
- Text alone is insufficient
- Metadata (sleep, stress, energy) provides critical context
- A strong system must handle irreducible label noise honestly

---

## Feature Engineering

### Text Features
- TF-IDF vectorizer with 300 features
- Unigrams + bigrams (captures "felt heavy", "okay session")
- min_df=2 to remove noise
- sublinear_tf=True to dampen high frequency terms
- Sentence Transformer (all-MiniLM-L6-v2) — 384-dim semantic embeddings

### Metadata Features
- sleep_hours (continuous, median imputed)
- energy_level (1-5 scale)
- stress_level (1-5 scale)
- duration_min (session length)
- time_of_day (ordinal encoded)
- reflection_quality (ordinal encoded)
- previous_day_mood (label encoded)
- ambience_type (one-hot encoded)
- face_emotion_hint (one-hot encoded, missing → "unknown")

### Engineered Features
- sleep_deprived (binary: sleep < 5 hours)
- high_stress_low_energy (binary: stress ≥ 4 AND energy ≤ 2)
- stress_energy_ratio (stress / energy+1)
- text_word_count (number of words in reflection)
- is_short_text (binary: word count ≤ 3)

### Combined Matrix
- TF-IDF pipeline: 324 features (300 TF-IDF + 24 metadata)
- Ensemble pipeline: 408 features (384 embeddings + 24 metadata)

---

## Model Choice

### Emotional State Classification
- Algorithm: XGBoost Classifier
- Tuning: RandomizedSearchCV (20 iterations, 5-fold CV)
- Cross-val Accuracy: ~51% (ensemble pipeline)
- Note: 51% on a noisy 6-class problem is 3x better than random (16.7%)

### Intensity Prediction
- Algorithm: XGBoost Classifier (ordinal treated as classification)
- Labels shifted -1 for XGBoost compatibility (1-5 → 0-4)

### Text Representation
- TF-IDF: 300 features, unigrams + bigrams
- Sentence Transformer: all-MiniLM-L6-v2, 384-dim embeddings
- Final pipeline uses sentence embeddings + metadata (408 features)

### Why not larger transformers?
- Average text length is 11 words — short texts limit transformer benefit
- all-MiniLM-L6-v2 is lightweight (~90MB) and runs on-device
- Larger models gave marginal gains on this noisy dataset

---

## Decision Engine

Rule-based logic with score-based tie breaking.

### What to do
Based on predicted state + intensity band (low/high):

| State | Low Intensity | High Intensity |
|---|---|---|
| calm | sound_therapy | light_planning |
| focused | light_planning | light_planning |
| neutral | movement | grounding |
| restless | grounding | yoga |
| mixed | pause | grounding |
| overwhelmed | rest | yoga |

Overrides applied when:
- stress ≥ 4 → shift to calming activities
- energy ≤ 2 → shift to rest/pause

### When to do it
Based on time of day + stress level band.

---

## Uncertainty Modeling

### Confidence Score
- Classifier: max class probability from predict_proba()
- Combined: average of state and intensity confidence

### Uncertain Flag
- Flag = 1 if state confidence < 0.40
- Additional flags for short text, missing values, contradictory signals

### Robustness Layer
- Short text (≤2 words) → confidence capped at 0.30
- Missing key fields → confidence reduced by 20%
- High stress + high energy → confidence reduced by 15%

---

## Ablation Study Results

| Model | Cross-val Accuracy |
|---|---|
| Text Only (TF-IDF) | 52.0% |
| Metadata Only | 17.8% |
| Text + Metadata (TF-IDF) | 49.7% |
| Sentence Embeddings only | 53.2% |
| Ensemble (TF-IDF + Embeddings + Metadata) | 53.7% |

Text is the primary signal. Metadata alone cannot predict emotional state.
Sentence embeddings outperform TF-IDF on short emotional texts.
Final improvement over baseline: +4%.

---

## Output Format

`outputs/predictions.csv` contains:

| Column | Description |
|---|---|
| id | User ID |
| predicted_state | Emotional state (6 classes) |
| predicted_intensity | Intensity level (1-5) |
| confidence | Combined confidence score (0-1) |
| uncertain_flag | 1 if prediction is uncertain |
| what_to_do | Recommended activity |
| when_to_do | Timing recommendation |
| supportive_message | Human-like guidance message |

---

## API

FastAPI server with auto-documentation.

### Start server
```bash
uvicorn api.app:app --reload
```

### Endpoints
| Endpoint | Method | Description |
|---|---|---|
| / | GET | Health check |
| /predict | POST | Full prediction pipeline |
| /health | GET | Server status |
| /docs | GET | Interactive API documentation |

### Example request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "journal_text": "felt heavy and tired today",
    "sleep_hours": 5,
    "energy_level": 2,
    "stress_level": 4,
    "time_of_day": "night"
  }'
```

---

## Limitations and Honest Observations

1. ~51% cross-val accuracy reflects genuine label noise, not model failure
2. Short texts carry insufficient signal for reliable prediction
3. Same text maps to different emotions — irreducible uncertainty
4. Metadata alone is not predictive of emotional state
5. early_morning has only 25 samples — rare class problem

---

## Future Improvements

- Better sleep deprivation feature engineering
- Conflict detection layer for contradictory text + metadata
- Federated learning for on-device personalization
- Merge early_morning → morning to fix rare class problem
- Larger sentence transformer for richer embeddings
- Label noise handling techniques