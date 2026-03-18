# Decision Engine — What + When
# Logic: Rules first, scoring to break ties

# --- Activity scoring weights ---
# Each activity has scores for different signals
# Higher score = more appropriate for that signal

from email.mime import message


ACTIVITY_SCORES = {
    'grounding':      {'stress': 2, 'energy': 1, 'restless': 2, 'mixed': 2, 'overwhelmed': 1},
    'yoga':           {'stress': 2, 'energy': 2, 'restless': 2, 'overwhelmed': 2, 'mixed': 1},
    'sound_therapy':  {'stress': 1, 'energy': 0, 'calm': 2, 'neutral': 1},
    'light_planning': {'stress': 0, 'energy': 2, 'focused': 2, 'calm': 1, 'neutral': 1},
    'rest':           {'stress': 1, 'energy': 0, 'overwhelmed': 2, 'calm': 1},
    'movement':       {'stress': 1, 'energy': 2, 'neutral': 2, 'focused': 1},
    'pause':          {'stress': 2, 'energy': 0, 'mixed': 2, 'overwhelmed': 1, 'restless': 1},
}

# --- Primary rules: state + intensity ---
PRIMARY_RULES = {
    ('calm',        'low'):  'sound_therapy',
    ('calm',        'high'): 'light_planning',
    ('focused',     'low'):  'light_planning',
    ('focused',     'high'): 'light_planning',
    ('neutral',     'low'):  'movement',
    ('neutral',     'high'): 'grounding',
    ('restless',    'low'):  'grounding',
    ('restless',    'high'): 'yoga',
    ('mixed',       'low'):  'pause',
    ('mixed',       'high'): 'grounding',
    ('overwhelmed', 'low'):  'rest',
    ('overwhelmed', 'high'): 'yoga',
}

# --- When rules: time of day + stress ---
WHEN_RULES = {
    ('early_morning', 'low'):  'within_15_min',
    ('early_morning', 'high'): 'now',
    ('morning',       'low'):  'within_15_min',
    ('morning',       'high'): 'now',
    ('afternoon',     'low'):  'later_today',
    ('afternoon',     'high'): 'within_15_min',
    ('evening',       'low'):  'tonight',
    ('evening',       'high'): 'tonight',
    ('night',         'low'):  'tonight',
    ('night',         'high'): 'now',
}


def get_intensity_band(intensity):
    """Convert numeric intensity to low/high band."""
    if intensity <= 2.5:
        return 'low'
    else:
        return 'high'


def get_stress_band(stress):
    """Convert numeric stress to low/high band."""
    if stress <= 2.5:
        return 'low'
    else:
        return 'high'


def score_activities(state, intensity, stress, energy, candidates):
    """
    Score candidate activities based on current signals.
    Used to break ties when primary rule needs refinement.
    """
    scores = {}
    for activity in candidates:
        score = 0
        weights = ACTIVITY_SCORES.get(activity, {})

        # State match bonus
        if state in weights:
            score += weights[state]

        # High stress → prefer calming activities
        if stress >= 4:
            score += weights.get('stress', 0)

        # Low energy → prefer rest/pause
        if energy <= 2:
            score += weights.get('energy', 0) * -1  # penalize high energy activities
        else:
            score += weights.get('energy', 0)

        scores[activity] = score

    # Return activity with highest score
    return max(scores, key=scores.get)


def decide(state, intensity, stress, energy, time_of_day):
    """
    Main decision function.
    Returns: what_to_do, when_to_do

    Logic:
    1. Get primary rule based on state + intensity band
    2. Refine with scoring if signals conflict
    3. Get timing from time_of_day + stress band
    """

    intensity_band = get_intensity_band(intensity)
    stress_band = get_stress_band(stress)

    # --- Step 1: Primary rule ---
    what = PRIMARY_RULES.get((state, intensity_band), 'pause')

    # --- Step 2: Refinement via scoring ---
    # If stress is very high and activity is not calming — override
    if stress >= 4 and what in ['light_planning', 'movement']:
        candidates = ['grounding', 'pause', 'rest', 'yoga']
        what = score_activities(state, intensity, stress, energy, candidates)

    # If energy is very low and activity is active — override
    if energy <= 2 and what in ['movement', 'yoga', 'light_planning']:
        candidates = ['rest', 'pause', 'sound_therapy', 'grounding']
        what = score_activities(state, intensity, stress, energy, candidates)

    # --- Step 3: When rule ---
    when = WHEN_RULES.get((time_of_day, stress_band), 'within_15_min')

    # Night override — always tonight or now, never later_today
    if time_of_day == 'night' and when == 'later_today':
        when = 'tonight'

    return what, when


def generate_message(state, intensity, what, when):
    """
    Generate a short supportive human-like message.
    """
    intensity_band = get_intensity_band(intensity)

    templates = {
        ('calm', 'low'):        "You seem calm and settled. A gentle sound session can deepen that peace.",
        ('calm', 'high'):       "You're in a good headspace. Use this clarity for some light planning.",
        ('focused', 'low'):     "You're in a focused state. A little planning now can go a long way.",
        ('focused', 'high'):    "Great focus energy. Channel it into structured planning.",
        ('neutral', 'low'):     "Things feel neutral right now. A short movement break can shift your energy.",
        ('neutral', 'high'):    "You seem unsettled beneath the surface. Grounding can help center you.",
        ('restless', 'low'):    "There's some restlessness here. A grounding exercise can slow things down.",
        ('restless', 'high'):   "You seem quite restless. Yoga can help release that tension.",
        ('mixed', 'low'):       "Your feelings seem mixed right now. Pausing and breathing can bring clarity.",
        ('mixed', 'high'):      "A lot seems to be swirling. Let's ground you first before anything else.",
        ('overwhelmed', 'low'): "You seem a little overwhelmed. Rest is the kindest thing right now.",
        ('overwhelmed', 'high'):"Things feel heavy right now. Yoga or rest — let your body lead.",
    }

    message = templates.get((state, intensity_band),
        "Take a moment for yourself. A short mindful activity can help reset your state.")

    when_text = {
        'now':              'Start right now.',
        'within_15_min':    'Try to do this within the next 15 minutes.',
        'later_today':      'Plan to do this later today.',
        'tonight':          'Save this for tonight.',
        'tomorrow_morning': 'Begin tomorrow morning with this.'
    }

    activity_text = what.replace('_', ' ')
    return f"{message} {when_text.get(when, '')} Activity: {activity_text}."


if __name__ == "__main__":
    # Quick test cases
    test_cases = [
        ('restless',    4, 4, 2, 'morning'),
        ('calm',        2, 1, 4, 'afternoon'),
        ('overwhelmed', 5, 5, 1, 'night'),
        ('focused',     3, 2, 4, 'morning'),
        ('mixed',       3, 4, 2, 'evening'),
    ]

    print(f"{'State':<12} {'Int':<5} {'Stress':<7} {'Energy':<7} {'Time':<15} {'What':<15} {'When':<20} Message")
    print("-" * 120)

    for state, intensity, stress, energy, time in test_cases:
        what, when = decide(state, intensity, stress, energy, time)
        msg = generate_message(state, intensity, what, when)
        print(f"{state:<12} {intensity:<5} {stress:<7} {energy:<7} {time:<15} {what:<15} {when:<20} {msg}")