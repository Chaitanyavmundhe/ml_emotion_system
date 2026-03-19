"""
ml_emotion_system — Full Pipeline (Ensemble)
Run this file to execute the complete end-to-end pipeline:
1. Preprocess raw data
2. Train base models
3. Generate predictions using ensemble (TF-IDF + Sentence Transformers + Metadata)
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import preprocess
from embeddings import predict_with_ensemble


def main():
    print("=" * 60)
    print("   ML Emotion System — Full Ensemble Pipeline")
    print("=" * 60)

    # --- Step 1: Preprocess ---
    print("\n[1/2] Preprocessing data...")
    preprocess(
        train_path='data/raw/train.csv',
        test_path='data/raw/test.csv',
        output_dir='data/processed/'
    )

    # --- Step 2: Predict with ensemble ---
    print("\n[2/2] Generating ensemble predictions...")
    results = predict_with_ensemble(
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