import pandas as pd
import re
import os

def load_data(train_path, test_path):
    """Load raw train and test CSV files."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def clean_text(text):
    """Clean journal text — lowercase, remove punctuation, strip whitespace."""
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
    return text


def handle_missing(df):
    """
    Handle missing values:
    - sleep_hours → fill with median
    - previous_day_mood → fill with mode
    - face_emotion_hint → fill with 'unknown' (distinct from 'none')
    """
    df['sleep_hours'] = df['sleep_hours'].fillna(df['sleep_hours'].median())
    
    if df['previous_day_mood'].isnull().sum() > 0:
        df['previous_day_mood'] = df['previous_day_mood'].fillna(
            df['previous_day_mood'].mode()[0]
        )
    
    df['face_emotion_hint'] = df['face_emotion_hint'].fillna('unknown')
    
    return df


def encode_categoricals(df):
    """
    Encode categorical columns to numeric using mapping.
    Keeps original columns and adds encoded versions.
    """
    # Time of day — ordered by energy cycle
    time_map = {
        'early_morning': 0,
        'morning': 1,
        'afternoon': 2,
        'evening': 3,
        'night': 4
    }

    # Reflection quality — ordered by clarity
    quality_map = {
        'vague': 0,
        'conflicted': 1,
        'clear': 2
    }

    # Previous day mood — same as emotional state labels
    mood_map = {
        'calm': 0,
        'focused': 1,
        'neutral': 2,
        'mixed': 3,
        'restless': 4,
        'overwhelmed': 5
    }

    df['time_of_day_enc'] = df['time_of_day'].map(time_map)
    df['reflection_quality_enc'] = df['reflection_quality'].map(quality_map)
    df['previous_day_mood_enc'] = df['previous_day_mood'].map(mood_map)

    # One-hot encode ambience and face_emotion_hint
    df = pd.get_dummies(df, columns=['ambience_type', 'face_emotion_hint'])

    return df


def preprocess(train_path, test_path, output_dir):
    """Full preprocessing pipeline."""
    
    # Step 1 — Load
    train, test = load_data(train_path, test_path)
    print(f"Loaded train: {train.shape}, test: {test.shape}")

    # Step 2 — Handle missing values
    train = handle_missing(train)
    test = handle_missing(test)
    print("Missing values handled.")

    # Step 3 — Clean text
    train['cleaned_text'] = train['journal_text'].apply(clean_text)
    test['cleaned_text'] = test['journal_text'].apply(clean_text)
    print("Text cleaned.")

    # Step 4 — Encode categoricals
    train = encode_categoricals(train)
    test = encode_categoricals(test)
    print("Categoricals encoded.")

    # Step 5 — Align columns (test may differ after get_dummies)
    train, test = train.align(test, join='left', axis=1, fill_value=0)
    print("Columns aligned between train and test.")

    # Step 6 — Save
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)
    print(f"Saved processed files to {output_dir}")

    return train, test


if __name__ == "__main__":
    preprocess(
        train_path='data/raw/train.csv',
        test_path='data/raw/test.csv',
        output_dir='data/processed/'
    )