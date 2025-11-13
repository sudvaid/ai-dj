import json
import itertools
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from src.rules import score_transition

# -----------------------------
# Dataset
# -----------------------------
class TrackPairDataset(Dataset):
    def __init__(self, features_file="data/features.json"):
        self.data = json.load(open(features_file))
        self.pairs, self.scores = self.prepare_pairs()

    def prepare_pairs(self):
        pairs, scores = [], []
        for a, b in itertools.combinations(self.data, 2):
            # Use actual rule-based score as regression target
            score = score_transition(a, b)
            features = np.concatenate([
                [a["bpm"], a["key"], a["energy"], a["spectral_centroid"]],
                [b["bpm"], b["key"], b["energy"], b["spectral_centroid"]]
            ])
            pairs.append(features)
            scores.append(score)
        return np.array(pairs, dtype=np.float32), np.array(scores, dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.scores[idx]

# -----------------------------
# Neural Network Model
# -----------------------------
class TransitionNN(nn.Module):
    def __init__(self, input_size=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Regression output
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Training Function
# -----------------------------
def train_model(features_file="data/features.json", epochs=100, batch_size=16, lr=0.001):
    # Load dataset
    dataset = TrackPairDataset(features_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = TransitionNN()
    criterion = nn.MSELoss()  # Regression loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    print("Neural network trained successfully")
    return model

# -----------------------------
# Optional: Quick test
# -----------------------------
if __name__ == "__main__":
    model = train_model()
    # Example: predict score between first two tracks
    dataset = TrackPairDataset()
    features = torch.tensor(dataset.pairs[0], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        score = model(features.unsqueeze(0)).item()
    print(f"Predicted score for first pair: {score:.2f}")
