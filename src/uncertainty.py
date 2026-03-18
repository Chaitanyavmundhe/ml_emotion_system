import numpy as np

# Confidence threshold below which we flag as uncertain
CONFIDENCE_THRESHOLD = 0.40


def get_confidence(proba):
    """
    Extract confidence score from class probability array.
    Confidence = highest class probability.
    
    Args:
        proba: np.array of shape (n_samples, n_classes)
    Returns:
        confidences: np.array of shape (n_samples,)
    """
    return np.max(proba, axis=1)


def get_uncertain_flag(confidences, threshold=CONFIDENCE_THRESHOLD):
    """
    Flag predictions as uncertain if confidence is below threshold.
    
    Args:
        confidences: np.array of confidence scores
        threshold: float, below this = uncertain
    Returns:
        flags: np.array of 0 or 1
    """
    return (confidences < threshold).astype(int)


def get_intensity_confidence(y_pred, y_true=None):
    """
    For regression, confidence is based on how close prediction
    is to a whole number (since intensity is 1-5 integers).
    
    Closer to whole number = more confident.
    Further from whole number = less confident (borderline case).
    
    Args:
        y_pred: np.array of predicted intensity values
    Returns:
        confidences: np.array of confidence scores (0-1)
    """
    # Distance from nearest integer
    distance = np.abs(y_pred - np.round(y_pred))
    
    # Convert to confidence: 0 distance = 1.0 confidence, 0.5 distance = 0.0 confidence
    confidences = 1.0 - (distance * 2)
    
    return np.clip(confidences, 0, 1)


def summarize_uncertainty(confidences, flags):
    """
    Print a summary of uncertainty across all predictions.
    """
    print(f"\n--- Uncertainty Summary ---")
    print(f"Mean confidence:     {confidences.mean():.4f}")
    print(f"Min confidence:      {confidences.min():.4f}")
    print(f"Max confidence:      {confidences.max():.4f}")
    print(f"Uncertain predictions: {flags.sum()} / {len(flags)} ({flags.mean()*100:.1f}%)")


if __name__ == "__main__":
    # Simulate model probability outputs
    np.random.seed(42)

    # Case 1 — confident prediction
    confident_proba = np.array([[0.85, 0.05, 0.03, 0.03, 0.02, 0.02]])
    c1 = get_confidence(confident_proba)
    f1 = get_uncertain_flag(c1)
    print(f"Confident case   → confidence: {c1[0]:.2f}, uncertain_flag: {f1[0]}")

    # Case 2 — uncertain prediction
    uncertain_proba = np.array([[0.20, 0.18, 0.17, 0.17, 0.15, 0.13]])
    c2 = get_confidence(uncertain_proba)
    f2 = get_uncertain_flag(c2)
    print(f"Uncertain case   → confidence: {c2[0]:.2f}, uncertain_flag: {f2[0]}")

    # Case 3 — borderline prediction
    borderline_proba = np.array([[0.42, 0.30, 0.15, 0.08, 0.03, 0.02]])
    c3 = get_confidence(borderline_proba)
    f3 = get_uncertain_flag(c3)
    print(f"Borderline case  → confidence: {c3[0]:.2f}, uncertain_flag: {f3[0]}")

    # Case 4 — intensity confidence
    intensity_preds = np.array([1.1, 2.5, 3.9, 4.0, 2.3])
    i_conf = get_intensity_confidence(intensity_preds)
    for pred, conf in zip(intensity_preds, i_conf):
        print(f"Intensity pred: {pred:.1f} → confidence: {conf:.2f}")