# src/playlist_generator.py
import json
import numpy as np
import torch
from math import isfinite
from src.ml_model import train_model, TrackPairDataset

# -----------------------------
# Prepare and train model
# -----------------------------
# Load dataset to get tracks (TrackPairDataset loads data from data/features.json)
dataset = TrackPairDataset()
tracks_data = dataset.data

# Train neural network (adjust epochs if you want)
model = train_model(epochs=50, batch_size=32, lr=1e-3)
model.eval()

# Force CPU usage, just in case
device = torch.device("cpu")
model.to(device)

# -----------------------------
# Helper: compute NN score for a pair (a,b)
# -----------------------------
def pair_score(model, a, b):
    features = np.concatenate([
        [a["bpm"], a["key"], a["energy"], a["spectral_centroid"]],
        [b["bpm"], b["key"], b["energy"], b["spectral_centroid"]]
    ]).astype(np.float32)
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    # out may be tensor shape (1,1) or (1,), convert safely
    try:
        score = float(out.squeeze().cpu().item())
    except Exception:
        # fallback if something unexpected
        score = float(np.nan)
    return score

# -----------------------------
# Generate Playlist (greedy)
# -----------------------------
unused_tracks = tracks_data.copy()
if not unused_tracks:
    raise SystemExit("No tracks found in data/features.json")

# Choose a start track (you can randomize or choose by some metric)
current_track = unused_tracks.pop(0)
playlist = [current_track["filename"]]
transition_scores = []

while unused_tracks:
    candidate_scores = []
    # compute scores for all candidates
    for t in unused_tracks:
        s = pair_score(model, current_track, t)
        candidate_scores.append((t, s))

    # Filter out NaN or non-finite scores
    valid = [(t, s) for (t, s) in candidate_scores if isfinite(s)]
    if valid:
        # select the candidate with the highest score
        next_track, best_score = max(valid, key=lambda ts: ts[1])
    else:
        # fallback: no valid predictions — pick the first remaining track deterministically
        next_track = unused_tracks[0]
        best_score = float('nan')

    # Safety check: ensure next_track is not None
    if next_track is None:
        # should never happen, but handle it gracefully
        next_track = unused_tracks[0]
        best_score = float('nan')

    # add to playlist and remove from unused
    playlist.append(next_track["filename"])
    transition_scores.append((current_track["filename"], next_track["filename"], best_score))
    current_track = next_track
    unused_tracks.remove(next_track)

# -----------------------------
# Print Playlist with Scores
# -----------------------------
print("✅ Generated Neural-Network DJ Playlist with Transition Scores:")
for i, (a, b, score) in enumerate(transition_scores, 1):
    score_str = f"{score:.2f}" if isfinite(score) else "NaN"
    print(f"{i}: {a} -> {b} | Score: {score_str}")

print("\nFull Playlist Order:")
for i, track in enumerate(playlist, 1):
    print(f"{i}: {track}")
