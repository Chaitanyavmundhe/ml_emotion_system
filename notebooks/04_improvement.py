import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from scipy.sparse import hstack, csr_matrix
import sys
sys.path.append('src')

train = pd.read_csv('data/processed/train_processed.csv')

# Engineered features
train['sleep_deprived'] = (train['sleep_hours'] < 5).astype(int)
train['high_stress_low_energy'] = ((train['stress_level'] >= 4) & (train['energy_level'] <= 2)).astype(int)
train['stress_energy_ratio'] = train['stress_level'] / (train['energy_level'] + 1)
train['text_word_count'] = train['cleaned_text'].str.split().str.len()
train['is_short_text'] = (train['text_word_count'] <= 3).astype(int)

le = LabelEncoder()
y = le.fit_transform(train['emotional_state'])

xgb_params = {
    'n_estimators': 200, 'max_depth': 4,
    'learning_rate': 0.05, 'subsample': 0.7,
    'colsample_bytree': 0.7, 'random_state': 42,
    'eval_metric': 'mlogloss', 'verbosity': 0
}

# --- Step 1: TF-IDF ---
print("Building TF-IDF features...")
tfidf = TfidfVectorizer(max_features=300, ngram_range=(1,2), min_df=2, sublinear_tf=True)
X_tfidf = tfidf.fit_transform(train['cleaned_text'])
scores = cross_val_score(XGBClassifier(**xgb_params), X_tfidf, y, cv=5, scoring='accuracy')
print(f"TF-IDF Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# --- Step 2: Sentence Embeddings ---
print("\nBuilding sentence embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
X_emb = model.encode(list(train['cleaned_text']), batch_size=64, show_progress_bar=True)
scores = cross_val_score(XGBClassifier(**xgb_params), X_emb, y, cv=5, scoring='accuracy')
print(f"Embeddings Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# --- Step 3: Ensemble ---
print("\nBuilding ensemble...")
METADATA_COLS = ['sleep_hours', 'energy_level', 'stress_level', 'duration_min',
                 'time_of_day_enc', 'reflection_quality_enc', 'previous_day_mood_enc',
                 'sleep_deprived', 'high_stress_low_energy', 'stress_energy_ratio',
                 'text_word_count', 'is_short_text']
ambience_cols = [c for c in train.columns if c.startswith('ambience_type_')]
face_cols = [c for c in train.columns if c.startswith('face_emotion_hint_')]
all_meta = METADATA_COLS + ambience_cols + face_cols
X_meta = csr_matrix(train[all_meta].values.astype(float))
X_emb_sparse = csr_matrix(X_emb)

X_ensemble = hstack([X_tfidf, X_emb_sparse, X_meta])
print(f"Ensemble shape: {X_ensemble.shape}")
scores = cross_val_score(XGBClassifier(**xgb_params), X_ensemble, y, cv=5, scoring='accuracy')
print(f"Ensemble Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")