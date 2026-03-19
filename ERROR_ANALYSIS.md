# Error Analysis Report

## Overview
- Total training samples: 1200
- Total errors on training set: 148
- Overall training accuracy: 87.67%
- Cross-validation accuracy: ~50% (true generalization performance)

---

## Pattern Summary

| Pattern | Count | % of Errors |
|---|---|---|
| Conflicted reflection quality | 57 | 38.5% |
| Vague reflection quality | 52 | 35.1% |
| Short texts (≤5 words) | ~80 | ~54% |
| Mixed state misclassified | 31 | 20.9% |
| Calm state misclassified | 29 | 19.6% |

---

## Failure Case Analysis (10 Cases)

---

### Case 1 — Same Text, Different Labels
**Text:** "okay session"
**True Labels seen:** focused, neutral, restless, mixed (all for same text)
**Predicted:** calm

**What went wrong:**
The model learned one pattern for "okay session" and applies it everywhere.
But this text appears with 4 different true labels in training data.
No model can correctly learn this — it is a label noise problem.

**Why it failed:**
Identical input, contradictory supervision signal. The model averages out and picks the most frequent class it saw for this text.

**How to improve:**
Rely less on text for ambiguous short phrases. Weight metadata more heavily when text length < 5 words.

---

### Case 2 — High Confidence, Wrong Prediction
**Text:** "more clear today"
**True Label:** mixed
**Predicted:** calm (confidence: 0.51)

**What went wrong:**
"More clear" strongly suggests calm or focused in the training data.
Model is confidently wrong because the text pattern is misleading.

**Why it failed:**
The word "clear" is semantically associated with calm/focused states.
But this user meant emotional clarity after a mixed session — not calmness.

**How to improve:**
Context window matters. "More clear" implies a change from a previous state — previous_day_mood feature should have flagged this. Better feature interaction needed.

---

### Case 3 — Conflicting Signals
**Text:** "By the end felt heavy."
**True Label:** neutral
**Predicted:** overwhelmed (confidence: 0.49)

**What went wrong:**
"Felt heavy" is a strong overwhelmed signal in the training data.
But stress=1, energy=2, sleep=7 suggest a calm, rested person.
The text contradicts the metadata completely.

**Why it failed:**
Model weighted text signal over metadata signal.
A truly heavy person would have high stress — but stress is 1 here.

**How to improve:**
When text and metadata contradict each other, flag as uncertain and reduce confidence. Add a conflict detection layer.

---

### Case 4 — Misleading Positive Text
**Text:** "helped me plan my day"
**True Label:** overwhelmed
**Predicted:** focused (confidence: 0.44)

**What went wrong:**
"Helped me plan" is a focused/productive phrase.
But this person is actually overwhelmed — they used the session to cope, not to thrive.

**Why it failed:**
Surface-level text sentiment is positive but emotional state is negative.
The model cannot detect this irony without deeper context.

**How to improve:**
Sentiment analysis as a separate feature. Detect mismatch between positive text and high stress/low energy metadata.

---

### Case 5 — Rare Time of Day
**Text:** any text with early_morning
**True Label:** various
**Predicted:** various wrong

**What went wrong:**
early_morning has only 25 samples in training data.
Model has almost no examples to learn from for this time slot.

**Why it failed:**
Data sparsity for rare category. 25 samples vs 300 for other time slots.

**How to improve:**
Merge early_morning with morning for modeling purposes.
Or oversample early_morning entries during training.

---

### Case 6 — Vague Text with No Signal
**Text:** "it was fine"
**True Label:** appears 6 times with different labels
**Predicted:** wrong every time

**What went wrong:**
"It was fine" carries zero emotional signal.
It is a non-committal phrase used across all emotional states.

**Why it failed:**
No text feature can extract meaning from content-free phrases.
This is irreducible noise — no model can solve this without more context.

**How to improve:**
For texts with word count ≤ 3, ignore text entirely and rely only on metadata.
Flag these as high uncertainty automatically.

---

### Case 7 — Contradictory Metadata
**Text:** "back to normal after"
**True Label:** calm
**Predicted:** restless (confidence: 0.35)
**Signals:** stress=4, energy=3, sleep=4

**What went wrong:**
Text says "back to normal" — suggests calm resolution.
But stress=4 and sleep=4 suggest a tired, stressed person.
The model trusted the metadata stress signal over the text.

**Why it failed:**
Neither text nor metadata alone is reliable here.
The combination creates conflicting signals with no clear winner.

**How to improve:**
When stress and text sentiment conflict, default to uncertain flag.
Use reflection_quality=vague as an additional penalty on confidence.

---

### Case 8 — Emotional Ambiguity in Mixed State
**Text:** "Honestly felt lighter, not fully though."
**True Label:** mixed
**Predicted:** calm (confidence: 0.37)

**What went wrong:**
"Felt lighter" pulls toward calm.
"Not fully though" qualifies it — this is a mixed state by definition.
The model caught the positive part but missed the qualification.

**Why it failed:**
Negation and qualification handling is weak in TF-IDF.
"Not fully" as a bigram may not appear enough times to be learned.

**How to improve:**
Better text representations that capture negation — sentence transformers would handle this better than TF-IDF.

---

### Case 9 — Night Time Misclassification
**Text:** "Honestly helped me plan my day. Later it changed breathing slowed down."
**True Label:** restless
**Predicted:** calm (confidence: 0.43)
**Time:** night

**What went wrong:**
First half of text is positive — "helped me plan".
Second half reveals restlessness — "breathing slowed down".
Model likely averaged both signals and landed on calm.

**Why it failed:**
TF-IDF treats text as a bag of words — word order is lost.
"Helped" and "breathing slowed" are weighted equally with no temporal understanding.

**How to improve:**
Sequential models (LSTM, transformer) that read text left to right would catch the shift in emotional tone within the same text.

---

### Case 10 — Low Sleep, Wrong Prediction
**Text:** "more clear today"
**True Label:** overwhelmed
**Predicted:** calm (confidence: 0.42)
**Signals:** sleep=3.5, stress=2, energy=3

**What went wrong:**
Sleep=3.5 is the minimum in the dataset — severely sleep deprived.
Text says "more clear" which pulls strongly toward calm.
Model ignored the sleep deprivation signal completely.

**Why it failed:**
Sleep hours feature is continuous and the model did not learn that 3.5 is an extreme outlier.
A human would immediately flag 3.5 hours sleep as a red alert.

**How to improve:**
Create a binary feature: sleep_deprived = 1 if sleep_hours < 5.
This makes the extreme signal explicit and easier for the model to learn.

---

## Key Insights

1. **Text alone is insufficient** — 11-word average reflections carry too little signal
2. **Label noise is real** — same text appears with different labels, creating irreducible error
3. **Conflicted and vague reflections** account for 74% of errors
4. **Mixed state is hardest** — by definition it overlaps with every other state
5. **TF-IDF misses negation** — "not fully" and "not sure" are poorly handled
6. **Metadata contradictions** need explicit conflict detection
7. **Short texts (≤5 words)** should trigger automatic high uncertainty flag
8. **Sleep deprivation** needs to be an explicit engineered feature
9. **early_morning** is a data sparsity problem — needs oversampling or merging
10. **Sequential text understanding** (transformers) would meaningfully improve performance

---

## Recommended Improvements

| Improvement | Expected Impact |
|---|---|
| sleep_deprived binary feature | Catch extreme outliers |
| Merge early_morning → morning | Fix rare class problem |
| Text length uncertainty boost | Auto-flag short inputs |
| Conflict detection layer | Handle text vs metadata contradiction |
| Sentence transformers | Better text representation |
| Negation-aware features | Handle "not fully", "not sure" |

---

## Baseline Comparison

| Target | Metric | Baseline | Our Model | Improvement |
|---|---|---|---|---|
| Emotional State | Accuracy | 18.58% | 51.33% | +32.75% |
| Intensity | MAE | 1.19 | 1.23 | slightly worse |

### Key finding:
Emotional state prediction is meaningful — 2.76x better than random.
Intensity prediction is weak — slightly worse than always predicting the mean.
A production system should treat intensity as low confidence and rely
more on user self-reporting for intensity calibration.