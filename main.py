"""
ml_emotion_system — Full Pipeline
Run this file to execute the complete end-to-end pipeline:
1. Preprocess raw data
2. Train models
3. Generate predictions
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import preprocess
from train import train_emotion_classifier, train_intensity_regressor, train_intensity_classifier, evaluate_on_train
from predict import predict
from features import build_features
import pandas as pd


def main():
    print("=" * 60)
    print("   ML Emotion System — Full Pipeline")
    print("=" * 60)

    # --- Step 1: Preprocess ---
    print("\n[1/3] Preprocessing data...")
    preprocess(
        train_path='data/raw/train.csv',
        test_path='data/raw/test.csv',
        output_dir='data/processed/'
    )

    # --- Step 2: Train ---
    print("\n[2/3] Training models...")
    train = pd.read_csv('data/processed/train_processed.csv')
    test = pd.read_csv('data/processed/test_processed.csv')

    X_train, X_test, y_state, y_intensity, feature_names = build_features(
        train, test, save_dir='models/'
    )

    classifier, le = train_emotion_classifier(X_train, y_state)
    regressor = train_intensity_regressor(X_train, y_intensity)
    intensity_clf = train_intensity_classifier(X_train, y_intensity)

    evaluate_on_train(classifier, regressor, le, X_train, y_state, y_intensity)

    # --- Step 3: Predict ---
    print("\n[3/3] Generating predictions...")
    results = predict(
        train_path='data/processed/train_processed.csv',
        test_path='data/processed/test_processed.csv',
        model_dir='models/',
        output_dir='outputs/'
    )

    print("\n" + "=" * 60)
    print("   Pipeline Complete!")
    print(f"   Predictions saved to outputs/predictions.csv")
    print(f"   Total predictions: {len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()