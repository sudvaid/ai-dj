import librosa
import numpy as np
import os
import json

def extract_features(filepath):
    y, sr = librosa.load(filepath)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key = int(np.argmax(np.mean(chroma, axis=1)))
    energy = float(np.mean(librosa.feature.rms(y=y)))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    return {
        "filename": os.path.basename(filepath),
        "bpm": float(tempo),
        "key": key,
        "energy": energy,
        "spectral_centroid": spectral_centroid
    }

def extract_all_from_folder(folder="data/tracks", output="data/features.json"):
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    features = []
    for file in os.listdir(folder):
        if file.endswith(".mp3") or file.endswith(".wav"):
            print(f"Extracting features from {file}")
            features.append(extract_features(os.path.join(folder, file)))
    with open(output, "w") as f:
        json.dump(features, f, indent=2)
    print(f"Saved features for {len(features)} tracks to {output}")

if __name__ == "__main__":
    extract_all_from_folder()
