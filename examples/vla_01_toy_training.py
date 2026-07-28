#!/usr/bin/env python3
"""
vla_01_toy_training.py
======================
Toy VLA Training Demo — VLA 0→1 Stage 2.

Learns a simple vision-language-action mapping on synthetic data:
  red   + "move left"  -> [-1, 0]
  blue  + "move right" -> [ 1, 0]
  green + "move up"    -> [ 0, 1]
  yellow+ "move down"  -> [ 0,-1]

This is the first closed learning loop for VLA:
  Stage 0: minimal_vla.py     (understand architecture, random weights)
  Stage 1: THIS FILE          (train on toy data, watch loss drop)
  Stage 2: vla_demo.py        (run real pretrained SmolVLA inference)
  Stage 3: fine-tuning        (see docs/13-vla-zero-to-one.md)

Dependencies: pip install torch matplotlib
Run: python vla_01_toy_training.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. Synthetic Task: Color + Language -> 2D Action
# ============================================================

VOCAB = {
    "move": 0, "left": 1, "right": 2, "up": 3, "down": 4,
}

COLOR_MAP = {
    "red":    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "blue":   np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "green":  np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "yellow": np.array([1.0, 1.0, 0.0], dtype=np.float32),
}

ACTION_MAP = {
    ("red",    "move left"):  np.array([-1.0,  0.0], dtype=np.float32),
    ("blue",   "move right"): np.array([ 1.0,  0.0], dtype=np.float32),
    ("green",  "move up"):    np.array([ 0.0,  1.0], dtype=np.float32),
    ("yellow", "move down"):  np.array([ 0.0, -1.0], dtype=np.float32),
}


def generate_synthetic_image(color_name, img_size=32, noise=0.05):
    """Generate a 32x32 image dominated by the given color with small noise."""
    base = COLOR_MAP[color_name].reshape(3, 1, 1)
    img = np.ones((3, img_size, img_size), dtype=np.float32) * base
    img += np.random.randn(3, img_size, img_size).astype(np.float32) * noise
    img = np.clip(img, 0.0, 1.0)
    return img


def tokenize_text(text, max_len=4):
    """Simple word-level tokenization."""
    words = text.lower().split()
    tokens = [VOCAB.get(w, 0) for w in words]
    tokens += [0] * (max_len - len(tokens))
    return tokens[:max_len]


def generate_dataset(n_per_task=50):
    """Generate synthetic (image, text, action) triplets."""
    images, texts, actions = [], [], []
    for (color, text), action in ACTION_MAP.items():
        for _ in range(n_per_task):
            images.append(generate_synthetic_image(color))
            texts.append(tokenize_text(text))
            actions.append(action)
    return (
        np.array(images, dtype=np.float32),
        np.array(texts, dtype=np.int64),
        np.array(actions, dtype=np.float32),
    )


# ============================================================
# 2. Toy VLA Model
# ============================================================

class TinyVLA(nn.Module):
    """Tiny VLA: CNN vision encoder + text embedding + fusion + action head."""

    def __init__(self, vocab_size=8, embed_dim=16, action_dim=2):
        super().__init__()
        # Vision encoder: 32x32 -> 8x8 -> 4x4 -> 2x2
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 8 -> 4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 4 -> 1
            nn.Flatten(),             # (B, 64)
        )
        # Language encoder
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim * 4, 32),
            nn.ReLU(),
        )
        # Fusion + action head
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )

    def forward(self, images, texts):
        z_img = self.vision(images)              # (B, 64)
        z_txt = self.text_embed(texts)           # (B, 4, embed_dim)
        z_txt = z_txt.flatten(1)                 # (B, 4*embed_dim)
        z_txt = self.text_proj(z_txt)            # (B, 32)
        z = torch.cat([z_img, z_txt], dim=-1)    # (B, 96)
        action = self.fusion(z)                  # (B, 2)
        return action


# ============================================================
# 3. Training Loop
# ============================================================

def train_toy_vla(epochs=40, lr=1e-3, batch_size=32, n_per_task=50, device="cpu"):
    print("=" * 60)
    print("Toy VLA Training Demo")
    print("=" * 60)
    print(f"\nGenerating synthetic dataset ({n_per_task} samples per task)...")

    images, texts, actions = generate_dataset(n_per_task)
    print(f"  Dataset: {len(images)} samples")
    print(f"  Image shape: {images[0].shape}")
    print(f"  Action shape: {actions[0].shape}")

    # Convert to torch tensors
    images_t = torch.FloatTensor(images).to(device)
    texts_t = torch.LongTensor(texts).to(device)
    actions_t = torch.FloatTensor(actions).to(device)

    # Model
    model = TinyVLA().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    print(f"\nTraining on {device} for {epochs} epochs...")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    losses = []
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(len(images_t))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(images_t), batch_size):
            idx = perm[i:i + batch_size]
            batch_img = images_t[idx]
            batch_txt = texts_t[idx]
            batch_act = actions_t[idx]

            pred = model(batch_img, batch_txt)
            loss = mse(pred, batch_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            marker = "***" if avg_loss < 0.1 else ""
            print(f"  Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} {marker}")

    print(f"\nFinal loss: {losses[-1]:.4f} (started from {losses[0]:.4f})")
    return model, losses


# ============================================================
# 4. Evaluation & Visualization
# ============================================================

def evaluate_and_visualize(model, losses, device="cpu"):
    print("\n" + "=" * 60)
    print("Evaluation: Predicted vs True Actions")
    print("=" * 60)

    tasks = [
        ("red",    "move left",  np.array([-1.0,  0.0])),
        ("blue",   "move right", np.array([ 1.0,  0.0])),
        ("green",  "move up",    np.array([ 0.0,  1.0])),
        ("yellow", "move down",  np.array([ 0.0, -1.0])),
    ]

    model.eval()
    results = []
    with torch.no_grad():
        for color, text, true_act in tasks:
            img = torch.FloatTensor(generate_synthetic_image(color)).unsqueeze(0).to(device)
            txt = torch.LongTensor(tokenize_text(text)).unsqueeze(0).to(device)
            pred = model(img, txt).squeeze(0).cpu().numpy()
            err = np.linalg.norm(pred - true_act)
            results.append((color, text, true_act, pred, err))
            status = "OK" if err < 0.15 else "FAIL"
            print(f"  {color:8s} + '{text:12s}' -> pred={pred.round(2)} true={true_act} err={err:.3f} [{status}]")

    # Plot
    out_dir = Path(__file__).parent.parent / "results" / "vla"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Toy VLA: Predicted vs True Actions", fontsize=14, fontweight="bold")

    for ax, (color, text, true_act, pred, err) in zip(axes.flat, results):
        ax.arrow(0, 0, true_act[0], true_act[1], head_width=0.08, color="green",
                 label="True", alpha=0.7, linewidth=2)
        ax.arrow(0, 0, pred[0], pred[1], head_width=0.08, color="blue",
                 label="Predicted", alpha=0.7, linewidth=2)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{color}: '{text}'\nerror={err:.3f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "toy_training.png", dpi=150)
    print(f"\n[Saved] {out_dir / 'toy_training.png'}")

    # Loss curve
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(losses, marker="o", markersize=3, color="#0d6efd")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Toy VLA Training Loss Curve")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0.1, color="green", linestyle="--", alpha=0.5, label="target threshold")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "toy_training_loss.png", dpi=150)
    print(f"[Saved] {out_dir / 'toy_training_loss.png'}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}\n")

    model, losses = train_toy_vla(device=device)
    evaluate_and_visualize(model, losses, device=device)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("  This demo showed:")
    print("    1. A tiny CNN vision encoder + text embedding + fusion + action head")
    print("    2. Training on synthetic (color, language) -> action pairs")
    print("    3. Loss dropping from random to near-zero")
    print("    4. Evaluation on held-out samples showing correct predictions")
    print("\n  Next step: run vla_demo.py --mode synthetic for real API demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
