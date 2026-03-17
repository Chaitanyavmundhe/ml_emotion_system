import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor
import joblib
import os

from features import build_features


def train_emotion_classifier(X_train, y_state, save_dir='models/'):
    """
    Train emotional state classifier.
    Compares Random Forest vs XGBoost and saves the better one.
    """
    print("\n--- Emotional State Classification ---")

    # Label encode target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_state)

    # --- Random Forest ---
    print("\nTraining Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_scores = cross_val_score(rf, X_train, y_encoded, cv=5, scoring='accuracy')
    print(f"RF Cross-val Accuracy: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")

    # --- XGBoost ---
    print("\nTraining XGBoost Classifier...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )
    xgb_scores = cross_val_score(xgb, X_train, y_encoded, cv=5, scoring='accuracy')
    print(f"XGB Cross-val Accuracy: {xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")

    # --- Pick best model ---
    if xgb_scores.mean() >= rf_scores.mean():
        print("\nXGBoost wins — using XGBoost for emotional state.")
        best_model = xgb
    else:
        print("\nRandom Forest wins — using Random Forest for emotional state.")
        best_model = rf

    # Train best model on full training data
    best_model.fit(X_train, y_encoded)

    # Save model and label encoder
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(save_dir, 'emotion_classifier.pkl'))
    joblib.dump(le, os.path.join(save_dir, 'label_encoder.pkl'))
    print(f"Emotion classifier saved to {save_dir}")

    return best_model, le


def train_intensity_regressor(X_train, y_intensity, save_dir='models/'):
    """
    Train intensity regressor.
    Compares Random Forest vs XGBoost and saves the better one.
    """
    print("\n--- Intensity Regression ---")

    # --- Random Forest ---
    print("\nTraining Random Forest Regressor...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_scores = cross_val_score(rf, X_train, y_intensity, cv=5, scoring='neg_mean_absolute_error')
    print(f"RF Cross-val MAE: {-rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")

    # --- XGBoost ---
    print("\nTraining XGBoost Regressor...")
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    xgb_scores = cross_val_score(xgb, X_train, y_intensity, cv=5, scoring='neg_mean_absolute_error')
    print(f"XGB Cross-val MAE: {-xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")

    # --- Pick best model (lower MAE is better) ---
    if -xgb_scores.mean() <= -rf_scores.mean():
        print("\nXGBoost wins — using XGBoost for intensity.")
        best_model = xgb
    else:
        print("\nRandom Forest wins — using Random Forest for intensity.")
        best_model = rf

    # Train best model on full training data
    best_model.fit(X_train, y_intensity)

    # Save model
    joblib.dump(best_model, os.path.join(save_dir, 'intensity_regressor.pkl'))
    print(f"Intensity regressor saved to {save_dir}")

    return best_model


def evaluate_on_train(classifier, regressor, le, X_train, y_state, y_intensity):
    """Quick sanity check — evaluate on full training data."""
    print("\n--- Training Set Evaluation (sanity check) ---")

    y_pred_state = le.inverse_transform(classifier.predict(X_train))
    print("\nEmotional State Classification Report:")
    print(classification_report(y_state, y_pred_state))

    y_pred_intensity = regressor.predict(X_train)
    mae = mean_absolute_error(y_intensity, y_pred_intensity)
    r2 = r2_score(y_intensity, y_pred_intensity)
    print(f"Intensity MAE: {mae:.4f}")
    print(f"Intensity R2 Score: {r2:.4f}")


if __name__ == "__main__":
    # Load processed data
    train = pd.read_csv('data/processed/train_processed.csv')
    test = pd.read_csv('data/processed/test_processed.csv')

    # Build features
    X_train, X_test, y_state, y_intensity, feature_names = build_features(train, test)

    # Train models
    classifier, le = train_emotion_classifier(X_train, y_state)
    regressor = train_intensity_regressor(X_train, y_intensity)

    # Sanity check
    evaluate_on_train(classifier, regressor, le, X_train, y_state, y_intensity)

    print("\n✅ Training complete. Models saved to models/")