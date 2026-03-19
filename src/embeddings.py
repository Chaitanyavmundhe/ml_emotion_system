"""
Sentence Transformer Embeddings
Replaces TF-IDF with semantic text embeddings using all-MiniLM-L6-v2
Model size: ~90MB | Embedding dim: 384
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import os

# Same metadata columns as features.py
METADATA_COLS = [
    'sleep_hours',
    'energy_level',
    'stress_level',
    'duration_min',
    'time_of_day_enc',
    'reflection_quality_enc',
    'previous_day_mood_enc',
    'sleep_deprived',
    'high_stress_low_energy',
    'stress_energy_ratio',
    'text_word_count',
    'is_short_text'
]


def get_sentence_embeddings(texts, model_name='all-MiniLM-L6-v2', save_dir='models/'):
    """
    Convert text list to sentence embeddings.
    Returns numpy array of shape (n_samples, 384)
    """
    print(f"Loading sentence transformer: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # Save model for inference
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, 'sentence_transformer.pkl'))
    print(f"Sentence transformer saved to {save_dir}")

    return embeddings


def build_embedding_features(train, test, save_dir='models/'):
    """
    Build feature matrix using sentence embeddings + metadata.
    Compare with TF-IDF baseline.
    """

    # --- Engineered features for train and test ---
    for df in [train, test]:
        df['sleep_deprived'] = (df['sleep_hours'] < 5).astype(int)
        df['high_stress_low_energy'] = (
            (df['stress_level'] >= 4) & (df['energy_level'] <= 2)
        ).astype(int)
        df['stress_energy_ratio'] = df['stress_level'] / (df['energy_level'] + 1)
        df['text_word_count'] = df['cleaned_text'].str.split().str.len()
        df['is_short_text'] = (df['text_word_count'] <= 3).astype(int)

    # --- Sentence embeddings ---
    all_texts = list(train['cleaned_text']) + list(test['cleaned_text'])
    all_embeddings = get_sentence_embeddings(all_texts, save_dir=save_dir)

    train_embeddings = all_embeddings[:len(train)]
    test_embeddings  = all_embeddings[len(train):]

    # --- Metadata ---
    ambience_cols = [c for c in train.columns if c.startswith('ambience_type_')]
    face_cols     = [c for c in train.columns if c.startswith('face_emotion_hint_')]
    all_meta_cols = METADATA_COLS + ambience_cols + face_cols
    all_meta_cols = [c for c in all_meta_cols if c in train.columns and c in test.columns]

    X_train_meta = train[all_meta_cols].values.astype(float)
    X_test_meta  = test[all_meta_cols].values.astype(float)

    # --- Combine embeddings + metadata ---
    X_train = np.hstack([train_embeddings, X_train_meta])
    X_test  = np.hstack([test_embeddings,  X_test_meta])

    print(f"Final shape — train: {X_train.shape}, test: {X_test.shape}")

    return X_train, X_test, all_meta_cols


def compare_tfidf_vs_embeddings(train):
    """
    Run ablation: TF-IDF vs Sentence Embeddings on train set.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    le = LabelEncoder()
    y  = le.fit_transform(train['emotional_state'])

    xgb_params = {
        'n_estimators': 200, 'max_depth': 4,
        'learning_rate': 0.05, 'subsample': 0.7,
        'colsample_bytree': 0.7, 'random_state': 42,
        'eval_metric': 'mlogloss', 'verbosity': 0
    }

    # --- TF-IDF baseline ---
    print("\nRunning TF-IDF baseline...")
    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1,2), min_df=2, sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(train['cleaned_text'])
    scores_tfidf = cross_val_score(XGBClassifier(**xgb_params), X_tfidf, y, cv=5, scoring='accuracy')
    print(f"TF-IDF Accuracy: {scores_tfidf.mean():.4f} (+/- {scores_tfidf.std():.4f})")

    # --- Sentence Embeddings ---
    print("\nRunning Sentence Embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X_emb = model.encode(list(train['cleaned_text']), batch_size=64, show_progress_bar=True)
    scores_emb = cross_val_score(XGBClassifier(**xgb_params), X_emb, y, cv=5, scoring='accuracy')
    print(f"Sentence Embeddings Accuracy: {scores_emb.mean():.4f} (+/- {scores_emb.std():.4f})")

    print(f"\nImprovement: {(scores_emb.mean() - scores_tfidf.mean())*100:.2f}%")

    return scores_tfidf.mean(), scores_emb.mean()

def predict_with_ensemble(train_path, test_path, model_dir='models/', output_dir='outputs/'):
    """
    Full prediction pipeline using ensemble features.
    """
    import joblib
    from recommend import decide, generate_message
    from uncertainty import get_confidence, get_uncertain_flag

    print("\nLoading data...")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    # Build ensemble features
    X_train, X_test, _ = build_embedding_features(train, test, save_dir=model_dir)

    # Load models
    print("Loading models...")
    classifier    = joblib.load(os.path.join(model_dir, 'emotion_classifier.pkl'))
    intensity_clf = joblib.load(os.path.join(model_dir, 'intensity_classifier.pkl'))
    le            = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))

    # Retrain classifier on ensemble features
    print("Retraining classifier on ensemble features...")
    le2 = LabelEncoder()
    y   = le2.fit_transform(train['emotional_state'])

    from sklearn.model_selection import cross_val_score
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7,
        random_state=42, eval_metric='mlogloss', verbosity=0
    )
    scores = cross_val_score(clf, X_train, y, cv=5, scoring='accuracy')
    print(f"Ensemble Cross-val Accuracy: {scores.mean():.4f}")

    clf.fit(X_train, y)
    joblib.dump(clf, os.path.join(model_dir, 'ensemble_classifier.pkl'))
    joblib.dump(le2, os.path.join(model_dir, 'ensemble_label_encoder.pkl'))

    # Predict
    print("\nPredicting...")
    y_pred         = clf.predict(X_test)
    predicted_state = le2.inverse_transform(y_pred)
    proba          = clf.predict_proba(X_test)
    state_conf     = get_confidence(proba)
    uncertain_flag = get_uncertain_flag(state_conf)

    # Retrain intensity classifier on ensemble features
    print("Retraining intensity classifier on ensemble features...")
    intensity_clf2 = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='mlogloss', verbosity=0
    )
    intensity_clf2.fit(X_train, train['intensity'] - 1)
    joblib.dump(intensity_clf2, os.path.join(model_dir, 'ensemble_intensity_classifier.pkl'))

    # Intensity predictions
    intensity_raw       = intensity_clf2.predict(X_test) + 1
    predicted_intensity = np.clip(intensity_raw, 1, 5).astype(int)

    # Combined confidence
    intensity_proba = intensity_clf2.predict_proba(X_test)
    intensity_conf  = get_confidence(intensity_proba)
    confidence      = np.round((state_conf + intensity_conf) / 2, 4)
    uncertain_flag  = (state_conf < 0.40).astype(int)

    # Decision engine

    # Decision engine
    what_list, when_list, message_list = [], [], []
    for i in range(len(test)):
        what, when = decide(
            predicted_state[i],
            predicted_intensity[i],
            test['stress_level'].iloc[i],
            test['energy_level'].iloc[i],
            test['time_of_day'].iloc[i]
        )
        message = generate_message(predicted_state[i], predicted_intensity[i], what, when)
        what_list.append(what)
        when_list.append(when)
        message_list.append(message)

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    results = pd.DataFrame({
        'id':                  test['id'],
        'predicted_state':     predicted_state,
        'predicted_intensity': predicted_intensity,
        'confidence':          confidence,
        'uncertain_flag':      uncertain_flag,
        'what_to_do':          what_list,
        'when_to_do':          when_list,
        'supportive_message':  message_list
    })

    output_path = os.path.join(output_dir, 'predictions.csv')
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    print(f"Total predictions: {len(results)}")
    print(f"\nEmotional State Distribution:")
    print(results['predicted_state'].value_counts())
    print(f"\nUncertain predictions: {uncertain_flag.sum()} / {len(results)}")

    return results


if __name__ == "__main__":
    train = pd.read_csv('data/processed/train_processed.csv')
    test  = pd.read_csv('data/processed/test_processed.csv')

    print("Comparing TF-IDF vs Sentence Embeddings...")
    tfidf_acc, emb_acc = compare_tfidf_vs_embeddings(train)

    print("\nGenerating final predictions with ensemble...")
    results = predict_with_ensemble(
        train_path='data/processed/train_processed.csv',
        test_path='data/processed/test_processed.csv',
        model_dir='models/',
        output_dir='outputs/'
    )
    print("\nFirst 5 predictions:")
    print(results.head())