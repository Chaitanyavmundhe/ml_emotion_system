"""
ML Emotion System — FastAPI Server
Run with: uvicorn api.app:app --reload
Docs at:  http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from recommend import decide, generate_message
from uncertainty import get_confidence, get_uncertain_flag, get_intensity_confidence

# --- Initialize app ---
app = FastAPI(
    title="ML Emotion System",
    description="Predicts emotional state and recommends meaningful actions.",
    version="1.0.0"
)

# Allow React UI to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load models at startup ---
print("Loading models...")
classifier    = joblib.load("models/ensemble_classifier.pkl")
intensity_clf = joblib.load("models/ensemble_intensity_classifier.pkl")
le            = joblib.load("models/ensemble_label_encoder.pkl")
tfidf         = joblib.load("models/tfidf_vectorizer.pkl")
st_model      = joblib.load("models/sentence_transformer.pkl")
print("Models loaded.")


# --- Request schema ---
class UserInput(BaseModel):
    journal_text: str = Field(..., example="felt a little heavy but okay session")
    sleep_hours: float = Field(6.0, ge=0, le=12, example=6.0)
    energy_level: int = Field(3, ge=1, le=5, example=3)
    stress_level: int = Field(3, ge=1, le=5, example=3)
    time_of_day: str = Field("morning", example="morning")
    previous_day_mood: Optional[str] = Field("neutral", example="neutral")
    reflection_quality: Optional[str] = Field("clear", example="clear")
    ambience_type: Optional[str] = Field("forest", example="forest")
    duration_min: Optional[float] = Field(15.0, example=15.0)
    face_emotion_hint: Optional[str] = Field("unknown", example="unknown")


# --- Response schema ---
class PredictionResponse(BaseModel):
    predicted_state: str
    predicted_intensity: int
    confidence: float
    uncertain_flag: int
    what_to_do: str
    when_to_do: str
    supportive_message: str


# --- Helper: build feature vector from single input ---
def build_single_feature(input: UserInput):
    # Encodings
    time_map = {
        'early_morning': 0, 'morning': 1,
        'afternoon': 2, 'evening': 3, 'night': 4
    }
    quality_map = {'vague': 0, 'conflicted': 1, 'clear': 2}
    mood_map = {
        'calm': 0, 'focused': 1, 'neutral': 2,
        'mixed': 3, 'restless': 4, 'overwhelmed': 5
    }

    # Sentence embeddings only (matches training pipeline)
    cleaned = input.journal_text.lower().strip()
    X_emb = st_model.encode([cleaned], convert_to_numpy=True)

    from scipy.sparse import hstack as sp_hstack, csr_matrix as csr
    X_text = csr(X_emb)

    # Metadata features
    meta = {
        'sleep_hours':            input.sleep_hours,
        'energy_level':           input.energy_level,
        'stress_level':           input.stress_level,
        'duration_min':           input.duration_min or 15.0,
        'time_of_day_enc':        time_map.get(input.time_of_day, 1),
        'reflection_quality_enc': quality_map.get(input.reflection_quality, 1),
        'previous_day_mood_enc':  mood_map.get(input.previous_day_mood, 2),
        # Ambience one-hot
        'ambience_type_cafe':     1 if input.ambience_type == 'cafe' else 0,
        'ambience_type_forest':   1 if input.ambience_type == 'forest' else 0,
        'ambience_type_mountain': 1 if input.ambience_type == 'mountain' else 0,
        'ambience_type_ocean':    1 if input.ambience_type == 'ocean' else 0,
        'ambience_type_rain':     1 if input.ambience_type == 'rain' else 0,
        # Face one-hot
        'face_emotion_hint_calm_face':    1 if input.face_emotion_hint == 'calm_face' else 0,
        'face_emotion_hint_happy_face':   1 if input.face_emotion_hint == 'happy_face' else 0,
        'face_emotion_hint_neutral_face': 1 if input.face_emotion_hint == 'neutral_face' else 0,
        'face_emotion_hint_none':         1 if input.face_emotion_hint == 'none' else 0,
        'face_emotion_hint_tense_face':   1 if input.face_emotion_hint == 'tense_face' else 0,
        'face_emotion_hint_tired_face':   1 if input.face_emotion_hint == 'tired_face' else 0,
        'face_emotion_hint_unknown':      1 if input.face_emotion_hint == 'unknown' else 0,
    }

    # Engineered features
    word_count = len(input.journal_text.split())
    sleep_dep  = 1 if input.sleep_hours < 5 else 0
    hs_le      = 1 if (input.stress_level >= 4 and input.energy_level <= 2) else 0
    se_ratio   = input.stress_level / (input.energy_level + 1)
    is_short   = 1 if word_count <= 3 else 0

    meta['sleep_deprived']         = sleep_dep
    meta['high_stress_low_energy'] = hs_le
    meta['stress_energy_ratio']    = se_ratio
    meta['text_word_count']        = word_count
    meta['is_short_text']          = is_short

    X_meta = csr(np.array(list(meta.values())).reshape(1, -1))
    X = sp_hstack([X_text, X_meta])

    return X


# --- Routes ---
@app.get("/")
def root():
    return {
        "message": "ML Emotion System API is running.",
        "docs": "Visit /docs for interactive API documentation."
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(input: UserInput):
    """
    Takes user journal text and contextual signals.
    Returns predicted emotional state, intensity, decision, and supportive message.
    """

    # Build features
    X = build_single_feature(input)

    # Predict state
    state_encoded = classifier.predict(X)[0]
    predicted_state = le.inverse_transform([state_encoded])[0]
    state_proba = classifier.predict_proba(X)
    state_confidence = float(np.max(state_proba))

    # Predict intensity
    intensity_raw = intensity_clf.predict(X)[0] + 1  # shift back
    predicted_intensity = int(np.clip(intensity_raw, 1, 5))
    intensity_proba = intensity_clf.predict_proba(X)
    intensity_confidence = float(np.max(intensity_proba))

    # Combined confidence
    confidence = round((state_confidence + intensity_confidence) / 2, 4)
    uncertain_flag = 1 if state_confidence < 0.40 else 0

    # Robustness — short text penalty
    word_count = len(input.journal_text.split())
    if word_count <= 2:
        confidence = min(confidence, 0.30)
        uncertain_flag = 1

    # Decision engine
    what, when = decide(
        predicted_state,
        predicted_intensity,
        input.stress_level,
        input.energy_level,
        input.time_of_day
    )

    # Supportive message
    message = generate_message(predicted_state, predicted_intensity, what, when)

    return PredictionResponse(
        predicted_state=predicted_state,
        predicted_intensity=predicted_intensity,
        confidence=confidence,
        uncertain_flag=uncertain_flag,
        what_to_do=what,
        when_to_do=when,
        supportive_message=message
    )


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}