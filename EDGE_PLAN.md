# Edge & Offline Deployment Plan

## Overview
This document explains how the ml_emotion_system would be deployed
on mobile devices and run completely offline without any cloud dependency.

---

## Current Model Size Analysis

| Component | Size | Notes |
|---|---|---|
| TF-IDF Vectorizer | ~2 MB | 300 features, sparse |
| XGBoost Classifier | ~5 MB | 200 trees, depth 4 |
| Intensity Classifier | ~3 MB | 100 trees, depth 3 |
| Label Encoder | <1 KB | 6 classes |
| **Total** | **~10 MB** | Fits comfortably on mobile |

---

## Target Deployment: Mobile (Android/iOS)

### Option 1 — ONNX Export (Recommended)
Convert trained models to ONNX format for cross-platform inference.

**Steps:**
1. Export XGBoost model to ONNX using `skl2onnx`
2. Run inference using ONNX Runtime Mobile
3. TF-IDF vectorizer converted to lookup table

**Benefits:**
- ONNX Runtime Mobile is ~1 MB
- Runs on CPU — no GPU needed
- Supported on both Android and iOS
- Inference time < 50ms per prediction

**Example conversion:**
```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 319]))]
onnx_model = convert_sklearn(classifier, initial_types=initial_type)

with open("models/emotion_classifier.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

---

### Option 2 — TensorFlow Lite
Convert to TFLite for Android deployment.

**Steps:**
1. Retrain a lightweight neural network in TensorFlow
2. Convert to TFLite with quantization
3. Deploy using TFLite Android SDK

**Model size after quantization:** ~1-2 MB
**Inference time:** <20ms

---

## Latency Analysis

| Step | Estimated Time | Notes |
|---|---|---|
| Text preprocessing | 2ms | Regex + lowercase |
| TF-IDF vectorization | 5ms | Lookup based |
| Emotion classification | 15ms | XGBoost inference |
| Intensity prediction | 10ms | XGBoost inference |
| Decision engine | 1ms | Pure rule-based |
| Uncertainty scoring | 1ms | Simple math |
| **Total** | **~34ms** | Well under 100ms target |

---

## On-Device Architecture
```
User Input (text + sliders)
        ↓
Text Preprocessor (regex, lowercase)
        ↓
TF-IDF Vectorizer (local lookup table)
        ↓
Feature Builder (text + metadata)
        ↓
ONNX Runtime (emotion + intensity)
        ↓
Decision Engine (pure Python rules)
        ↓
Uncertainty Module (confidence math)
        ↓
Output (state, intensity, what, when, message)
```

---

## Storage Requirements

| Asset | Size |
|---|---|
| ONNX models | ~8 MB |
| TF-IDF vocabulary | ~500 KB |
| Decision rules | <10 KB |
| App code | ~2 MB |
| **Total app size** | **~11 MB** |

Well within mobile app size limits (100MB typical limit).

---

## Offline Capabilities

| Feature | Offline? | Notes |
|---|---|---|
| Emotion prediction | ✅ Yes | Fully local |
| Intensity prediction | ✅ Yes | Fully local |
| Decision engine | ✅ Yes | Rule-based, no network |
| Uncertainty scoring | ✅ Yes | Math only |
| Supportive message | ✅ Yes | Template-based |
| Model updates | ❌ No | Requires download |

**100% core functionality works offline.**

---

## Tradeoffs

| Decision | Tradeoff |
|---|---|
| TF-IDF over transformers | Smaller size vs lower accuracy |
| XGBoost over deep learning | Faster inference vs less expressive |
| Rule-based decisions | Explainable vs less adaptive |
| ONNX over native | Cross-platform vs slight overhead |
| Templates for messages | Fast vs less personalized |

---

## Future Optimizations

| Optimization | Impact |
|---|---|
| Model quantization (INT8) | 4x size reduction |
| Vocabulary pruning | Remove rare TF-IDF tokens |
| On-device fine-tuning | Personalize to user over time |
| Federated learning | Improve model without sharing data |
| Lightweight transformer (DistilBERT) | Better text understanding at ~50MB |

---

## Privacy Considerations

- All inference runs on-device — no data leaves the phone
- Journal text never sent to any server
- User emotional state stays completely private
- No API keys or cloud dependencies required

---

## Recommended Stack

| Layer | Technology |
|---|---|
| Mobile framework | React Native / Flutter |
| ML inference | ONNX Runtime Mobile |
| Local storage | SQLite |
| Text processing | On-device Python (Chaquopy) or JS port |
| UI | Native components |