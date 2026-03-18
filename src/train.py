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
    Train emotional state classifier with hyperparameter tuning.
    """
    print("\n--- Emotional State Classification ---")

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import RandomizedSearchCV

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_state)

    # --- Tuned XGBoost ---
    print("\nTuning XGBoost Classifier...")
    xgb = XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_encoded)

    print(f"\nBest Params: {search.best_params_}")
    print(f"Best Cross-val Accuracy: {search.best_score_:.4f}")

    best_model = search.best_estimator_

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

def train_intensity_classifier(X_train, y_intensity, save_dir='models/'):
    """
    Train intensity as classifier instead of regressor.
    Handles ordinal nature better than regression on noisy data.
    """
    print("\n--- Intensity Classification ---")

    from sklearn.model_selection import RandomizedSearchCV

    xgb = XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_intensity-1)  # Shift to 0-4 for classification
    print("Note: intensity labels shifted by -1 for XGBoost (1-5 → 0-4)")

    print(f"\nBest Params: {search.best_params_}")
    print(f"Best Cross-val MAE: {-search.best_score_:.4f}")

    best_model = search.best_estimator_
    joblib.dump(best_model, os.path.join(save_dir, 'intensity_classifier.pkl'))
    print(f"Intensity classifier saved to {save_dir}")

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
    intensity_clf = train_intensity_classifier(X_train, y_intensity)

    # Sanity check
    evaluate_on_train(classifier, regressor, le, X_train, y_state, y_intensity)

    print("\n✅ Training complete. Models saved to models/")