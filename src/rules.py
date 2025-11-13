def harmonic_compatible(key_a, key_b):
    # Two keys are compatible if they are the same or 1 semitone apart
    return abs(key_a - key_b) <= 1

def score_transition(a, b):
    """
    a, b: dicts containing track features like bpm, key, energy
    Returns a simple score (higher = better transition)
    """
    score = 0

    # BPM match
    if abs(a["bpm"] - b["bpm"]) <= 10:
        score += 1

    # Harmonic key compatibility
    if harmonic_compatible(a["key"], b["key"]):
        score += 2

    # Energy continuity (avoid big drops)
    if b["energy"] >= a["energy"] * 0.9:
        score += 1

    return score
