# ML Emotion System

An intelligent emotion understanding and decision-making system that takes
user reflections and contextual signals to predict emotional states,
recommend meaningful actions, and guide users toward better mental states.

---

## Project Structure
```
ml_emotion_system/
├── data/
│   ├── raw/              # Original dataset (not tracked in git)
│   └── processed/        # Cleaned and transformed data
├── models/               # Saved trained models (not tracked in git)
├── notebooks/            # EDA and analysis notebooks
│   ├── 01_eda.ipynb
│   ├── 02_ablation_study.ipynb
│   └── 03_error_analysis.ipynb
├── outputs/              # Final predictions CSV
├── src/
│   ├── preprocess.py     # Data cleaning and encoding
│   ├── features.py       # TF-IDF + metadata feature engineering
│   ├── train.py          # Model training (XGBoost + Random Forest)
│   ├── predict.py        # Full prediction pipeline
│   ├── recommend.py      # Decision engine (what + when)
│   └── uncertainty.py    # Confidence scoring and uncertain flags
├── ERROR_ANALYSIS.md     # 10 failure case analysis
├── EDGE_PLAN.md          # Mobile and offline deployment plan
├── requirements.txt      # Python dependencies
└── main.py               # End-to-end pipeline entry point
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

### Step 1 — Preprocess
```bash
python src/preprocess.py
```

### Step 2 — Train models
```bash
python src/train.py
```

### Step 3 — Generate predictions
```bash
python src/predict.py
```

### Run full pipeline at once
```bash
python main.py
```

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

### Combined Matrix
319 total features = 300 TF-IDF + 19 metadata

---

## Model Choice

### Emotional State Classification
- Algorithm: XGBoost Classifier
- Tuning: RandomizedSearchCV (20 iterations, 5-fold CV)
- Cross-val Accuracy: ~50%
- Note: 50% on a noisy 6-class problem is 3x better than random (16.7%)

### Intensity Prediction
- Algorithm: XGBoost Classifier (ordinal treated as classification)
- Labels shifted -1 for XGBoost compatibility (1-5 → 0-4)
- Cross-val MAE: ~1.6

### Why not deep learning?
- Average text length is 11 words — too short for transformers to shine
- TF-IDF captures the available signal efficiently
- XGBoost is faster, more interpretable, and runs on-device

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
| Text Only | 52.0% |
| Metadata Only | 17.8% |
| Text + Metadata | 49.7% |

Text is the primary signal. Metadata alone cannot predict emotional state.
Combining them does not always help due to noise introduction.

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
| robustness_warning | Warning if input quality is low |

---

## Limitations and Honest Observations

1. ~50% cross-val accuracy reflects genuine label noise, not model failure
2. Short texts carry insufficient signal for reliable prediction
3. Same text maps to different emotions — irreducible uncertainty
4. Metadata alone is not predictive of emotional state
5. early_morning has only 25 samples — rare class problem

---

## Future Improvements

- Sentence transformers for richer text embeddings
- sleep_deprived binary feature for extreme outlier detection
- Conflict detection layer for contradictory text + metadata
- Federated learning for on-device personalization
- Merge early_morning → morning to fix rare class problem