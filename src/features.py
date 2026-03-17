import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib
import os


# Metadata columns we will use — chosen based on EDA insights
METADATA_COLS = [
    'sleep_hours',
    'energy_level',
    'stress_level',
    'duration_min',
    'time_of_day_enc',
    'reflection_quality_enc',
    'previous_day_mood_enc'
]


def build_features(train, test, tfidf_max_features=300, save_dir='models/'):
    """
    Build feature matrix by combining TF-IDF text features + metadata.
    
    Returns:
        X_train, X_test, y_state, y_intensity, feature_names
    """

    # --- Step 1: TF-IDF on cleaned text ---
    print("Building TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=(1, 2),   # unigrams + bigrams (e.g. "felt heavy")
        min_df=2,             # ignore very rare words
        sublinear_tf=True     # dampen high frequency terms
    )

    X_train_text = tfidf.fit_transform(train['cleaned_text'])
    X_test_text = tfidf.transform(test['cleaned_text'])
    print(f"TF-IDF shape — train: {X_train_text.shape}, test: {X_test_text.shape}")

    # --- Step 2: Metadata features ---
    print("Building metadata features...")
    
    # Get ambience and face_emotion_hint one-hot columns dynamically
    ambience_cols = [c for c in train.columns if c.startswith('ambience_type_')]
    face_cols = [c for c in train.columns if c.startswith('face_emotion_hint_')]
    
    all_meta_cols = METADATA_COLS + ambience_cols + face_cols
    
    # Keep only columns that exist in both train and test
    all_meta_cols = [c for c in all_meta_cols if c in train.columns and c in test.columns]

    X_train_meta = csr_matrix(train[all_meta_cols].values.astype(float))
    X_test_meta = csr_matrix(test[all_meta_cols].values.astype(float))
    print(f"Metadata shape — train: {X_train_meta.shape}, test: {X_test_meta.shape}")

    # --- Step 3: Combine text + metadata ---
    X_train = hstack([X_train_text, X_train_meta])
    X_test = hstack([X_test_text, X_test_meta])
    print(f"Combined shape — train: {X_train.shape}, test: {X_test.shape}")

    # --- Step 4: Target variables ---
    y_state = train['emotional_state']
    y_intensity = train['intensity']

    # --- Step 5: Feature names (for importance analysis later) ---
    tfidf_names = tfidf.get_feature_names_out().tolist()
    feature_names = tfidf_names + all_meta_cols

    # --- Step 6: Save TF-IDF vectorizer for inference ---
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(tfidf, os.path.join(save_dir, 'tfidf_vectorizer.pkl'))
    print(f"TF-IDF vectorizer saved to {save_dir}")

    return X_train, X_test, y_state, y_intensity, feature_names


if __name__ == "__main__":
    # Quick test
    train = pd.read_csv('data/processed/train_processed.csv')
    test = pd.read_csv('data/processed/test_processed.csv')

    X_train, X_test, y_state, y_intensity, feature_names = build_features(train, test)

    print(f"\nFinal X_train shape: {X_train.shape}")
    print(f"Final X_test shape: {X_test.shape}")
    print(f"Target classes: {y_state.unique()}")
    print(f"Intensity range: {y_intensity.min()} - {y_intensity.max()}")
    print(f"Total features: {len(feature_names)}")