import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add src to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features import build_features
from uncertainty import get_confidence, get_uncertain_flag, get_intensity_confidence, summarize_uncertainty
from recommend import decide, generate_message

def robustness_check(row):
    """
    Check input quality before prediction.
    Returns: warning flags dict

    Handles:
    - Very short text (word count <= 2)
    - Missing values in key fields
    - Contradictory signals (high energy + overwhelmed hint)
    """
    flags = {
        'short_text': False,
        'missing_values': False,
        'contradictory_signals': False,
        'robustness_warning': None
    }

    # --- Check 1: Very short text ---
    word_count = len(str(row.get('journal_text', '')).split())
    if word_count <= 2:
        flags['short_text'] = True
        flags['robustness_warning'] = "Very short text — prediction unreliable, defaulting to high uncertainty."

    # --- Check 2: Missing values ---
    key_fields = ['sleep_hours', 'energy_level', 'stress_level']
    for field in key_fields:
        if pd.isnull(row.get(field)):
            flags['missing_values'] = True
            flags['robustness_warning'] = f"Missing value in {field} — confidence reduced."
            break

    # --- Check 3: Contradictory signals ---
    energy = row.get('energy_level', 3)
    stress = row.get('stress_level', 3)
    face = row.get('face_emotion_hint', '')

    # High energy but tense face — contradiction
    if energy >= 4 and stress >= 4:
        flags['contradictory_signals'] = True
        flags['robustness_warning'] = "High energy + high stress detected — conflicting signals, uncertainty increased."

    # Very low sleep but calm face — contradiction
    sleep = row.get('sleep_hours', 6)
    if sleep <= 4 and face == 'calm_face':
        flags['contradictory_signals'] = True
        flags['robustness_warning'] = "Low sleep + calm face — contradictory signals detected."

    return flags


def apply_robustness_penalty(confidence, flags):
    """
    Reduce confidence score based on robustness flags.
    """
    if flags['short_text']:
        confidence = min(confidence, 0.30)  # cap at 30%

    if flags['missing_values']:
        confidence = confidence * 0.80  # reduce by 20%

    if flags['contradictory_signals']:
        confidence = confidence * 0.85  # reduce by 15%

    return round(confidence, 4)


def load_models(model_dir='models/'):
    """Load all saved models and encoders."""
    classifier = joblib.load(os.path.join(model_dir, 'emotion_classifier.pkl'))
    regressor = joblib.load(os.path.join(model_dir, 'intensity_regressor.pkl'))
    le = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    print("Models loaded successfully.")
    return classifier, regressor, le


def predict(train_path, test_path, model_dir='models/', output_dir='outputs/'):
    """
    Full prediction pipeline.
    Loads data → builds features → predicts → uncertainty → decision → saves CSV.
    """

    # --- Step 1: Load processed data ---
    print("\nLoading processed data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # --- Step 2: Build features ---
    print("Building features...")
    X_train, X_test, y_state, y_intensity, feature_names = build_features(train, test)

    # --- Step 3: Load models ---
    classifier, regressor, le = load_models(model_dir)

    # --- Step 4: Predict emotional state ---
    print("\nPredicting emotional state...")
    y_pred_encoded = classifier.predict(X_test)
    predicted_state = le.inverse_transform(y_pred_encoded)

    # Get probabilities for confidence scoring
    proba = classifier.predict_proba(X_test)
    state_confidence = get_confidence(proba)
    state_uncertain = get_uncertain_flag(state_confidence)

    # --- Step 5: Predict intensity ---
    print("Predicting intensity...")
    # Load intensity classifier
    intensity_clf = joblib.load(os.path.join(model_dir, 'intensity_classifier.pkl'))

    # Predict — add 1 back since we shifted labels during training
    predicted_intensity = intensity_clf.predict(X_test) + 1
    predicted_intensity = np.clip(predicted_intensity, 1, 5).astype(int)

    # Confidence from classifier probabilities
    intensity_proba = intensity_clf.predict_proba(X_test)
    intensity_confidence = get_confidence(intensity_proba)

    # --- Step 6: Combined confidence ---
    # Final confidence = average of state and intensity confidence
    combined_confidence = (state_confidence + intensity_confidence) / 2
    combined_confidence = np.round(combined_confidence, 4)

    # Flag uncertain only based on state confidence
    # Intensity is inherently noisy — don't double penalize
    uncertain_flag = (state_confidence < 0.40).astype(int)

    # --- Step 7: Decision engine ---
    print("Running decision engine...")
    what_list = []
    when_list = []
    message_list = []

    robustness_warnings = []

    for i in range(len(test)):
        state = predicted_state[i]
        intensity = predicted_intensity[i]
        stress = test['stress_level'].iloc[i]
        energy = test['energy_level'].iloc[i]
        time_of_day = test['time_of_day'].iloc[i]

        # Robustness check
        row = test.iloc[i].to_dict()
        flags = robustness_check(row)
        combined_confidence[i] = apply_robustness_penalty(
            combined_confidence[i], flags
        )

        # If short text or contradictory — mark as uncertain
        if flags['short_text'] or flags['contradictory_signals']:
            uncertain_flag[i] = 1

        robustness_warnings.append(flags['robustness_warning'])

        what, when = decide(state, intensity, stress, energy, time_of_day)
        message = generate_message(state, intensity, what, when)

        what_list.append(what)
        when_list.append(when)
        message_list.append(message)

    # --- Step 8: Build output dataframe ---
    results = pd.DataFrame({
        'id':                  test['id'],
        'predicted_state':     predicted_state,
        'predicted_intensity': predicted_intensity,
        'confidence':          combined_confidence,
        'uncertain_flag':      uncertain_flag,
        'what_to_do':          what_list,
        'when_to_do':          when_list,
        'supportive_message':  message_list,
        'robustness_warning':  robustness_warnings
    })

    # --- Step 9: Save predictions ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'predictions.csv')
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

    # --- Step 10: Summary ---
    print(f"\n--- Prediction Summary ---")
    print(f"Total predictions: {len(results)}")
    print(f"\nEmotional State Distribution:")
    print(results['predicted_state'].value_counts())
    print(f"\nIntensity Distribution:")
    print(results['predicted_intensity'].value_counts().sort_index())
    print(f"\nUncertain predictions: {uncertain_flag.sum()} / {len(results)}")
    print(f"Mean confidence: {combined_confidence.mean():.4f}")

    summarize_uncertainty(state_confidence, state_uncertain)

    return results


if __name__ == "__main__":
    results = predict(
        train_path='data/processed/train_processed.csv',
        test_path='data/processed/test_processed.csv',
        model_dir='models/',
        output_dir='outputs/'
    )

    print("\nFirst 5 predictions:")
    print(results.head())