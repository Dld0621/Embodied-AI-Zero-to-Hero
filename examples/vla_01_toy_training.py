#!/usr/bin/env python3
"""
vla_01_toy_training.py
=====================
Toy VLA Training Demo — VLA 0->1 Stage 2 (Dual-Target, True Fusion).

THE PROBLEM THIS FIXES
----------------------
An earlier version of this demo had a *shortcut*: color and text were
perfectly correlated (red + "move left", blue + "move right", ...). The
model could solve the task using *only* vision OR *only* language, so it
never had to actually fuse the two modalities. That is bad pedagogy for a
file whose entire point is "watch a VLA fuse vision + language".

THE NEW TASK: DUAL-TARGET
-------------------------
The image now contains **two** colored targets (one on the left, one on the
right). The language instruction selects **which** target to move to.

    Image: red at left, blue at right
        "move to red"  -> [-1, 0]   (go left,  to red)
        "move to blue" -> [ 1, 0]   (go right, to blue)

Both modalities are now strictly required:
  * The IMAGE determines WHERE each target is (left / right position).
  * The LANGUAGE determines WHICH target to select.

Crucially, the left/right assignment of colors is **randomized per sample**,
so the model cannot memorize "red = left". It must read the position off the
image every time. And because the same image is paired with two different
instructions (yielding opposite actions), and the same instruction is paired
with two different image layouts (yielding opposite actions), neither
modality alone can solve the task. True fusion is forced.

A modality-ablation study at the end verifies this empirically:
  1. Full model        (correct image + correct text)            -> high accuracy
  2. Text-swapped      (correct image + WRONG target text)       -> low accuracy
  3. Image-flipped     (image flipped left/right + correct text) -> low accuracy

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
# 1. Synthetic Task: Dual-Target Vision-Language -> 2D Action
# ============================================================

# 6 content tokens + <pad> token
VOCAB = {"<pad>": 0, "move": 1, "to": 2, "red": 3, "blue": 4, "green": 5, "yellow": 6}
VOCAB_SIZE = len(VOCAB)  # 7

COLOR_MAP = {
    "red":    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "blue":   np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "green":  np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "yellow": np.array([1.0, 1.0, 0.0], dtype=np.float32),
}

# The 6 unordered color pairs -> 6 base image layouts.
COLOR_PAIRS = [
    ("red", "blue"), ("red", "green"), ("red", "yellow"),
    ("blue", "green"), ("blue", "yellow"), ("green", "yellow"),
]
# 6 pairs x 2 selectable targets = 12 base (image_layout, target_color) patterns.
# For each pattern we sample many noisy images whose left/right color
# assignment is randomized, so the action is read off the actual image.

ACTION_LEFT = np.array([-1.0, 0.0], dtype=np.float32)   # target is on the left
ACTION_RIGHT = np.array([1.0, 0.0], dtype=np.float32)   # target is on the right

SEED = 42


def generate_dual_target_image(color_left, color_right, img_size=32,
                               square_size=8, noise=0.08, rng=None):
    """Render a 32x32 image with two colored squares.

    One square lives in the left half, the other in the right half. The
    horizontal (and vertical) positions are *randomized within each half* so
    the model cannot rely on fixed pixel coordinates -- only the left/right
    bin matters for the action, and that bin is what the CNN must preserve.
    """
    if rng is None:
        rng = np.random.default_rng()
    img = np.zeros((3, img_size, img_size), dtype=np.float32)  # (C, H, W)
    half = square_size // 2
    # Horizontal (column/x) center constrained to stay fully inside each half.
    cx_l = int(rng.integers(half, img_size // 2 - half + 1))            # left half
    cx_r = int(rng.integers(img_size // 2 + half, img_size - half + 1))  # right half
    cy = int(rng.integers(half, img_size - half + 1))                   # vertical center
    img[:, cy - half:cy + half, cx_l - half:cx_l + half] = COLOR_MAP[color_left].reshape(3, 1, 1)
    img[:, cy - half:cy + half, cx_r - half:cx_r + half] = COLOR_MAP[color_right].reshape(3, 1, 1)
    img += rng.standard_normal((3, img_size, img_size)).astype(np.float32) * noise
    img = np.clip(img, 0.0, 1.0)
    return img


def tokenize_text(text, max_len=4):
    """Simple word-level tokenization, zero-padded to max_len."""
    words = text.lower().split()
    tokens = [VOCAB.get(w, 0) for w in words]
    tokens = tokens + [0] * (max_len - len(tokens))
    return tokens[:max_len]


def generate_dataset(n_per_pattern=100, seed=SEED):
    """Generate (image, text, action) triplets for the dual-target task.

    Returns arrays plus per-sample metadata (color_left, color_right,
    target_color) which is needed later to build the modality-ablation sets.
    """
    rng = np.random.default_rng(seed)
    images, texts, actions, meta = [], [], [], []
    for (cA, cB) in COLOR_PAIRS:
        for target in (cA, cB):
            for _ in range(n_per_pattern):
                # Randomize which color sits on the left -> forces reading the image.
                if rng.random() < 0.5:
                    color_left, color_right = cA, cB
                else:
                    color_left, color_right = cB, cA
                img = generate_dual_target_image(color_left, color_right, rng=rng)
                text = f"move to {target}"
                tokens = tokenize_text(text)
                # Action = direction toward the selected target's side.
                action = ACTION_LEFT if color_left == target else ACTION_RIGHT
                images.append(img)
                texts.append(tokens)
                actions.append(action)
                meta.append((color_left, color_right, target))
    return (
        np.array(images, dtype=np.float32),
        np.array(texts, dtype=np.int64),
        np.array(actions, dtype=np.float32),
        meta,
    )


def split_dataset(images, texts, actions, meta, seed=SEED,
                  frac_train=0.6, frac_val=0.2):
    """Shuffle once (fixed seed) and split into train / val / test."""
    rng = np.random.default_rng(seed)
    n = len(images)
    idx = rng.permutation(n)
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)
    splits = {}
    for name, sl in [
        ("train", slice(0, n_train)),
        ("val", slice(n_train, n_train + n_val)),
        ("test", slice(n_train + n_val, n)),
    ]:
        s = idx[sl]
        splits[name] = {
            "images": images[s],
            "texts": texts[s],
            "actions": actions[s],
            "meta": [meta[i] for i in s],
        }
    return splits


# ============================================================
# 2. Toy VLA Model
# ============================================================

class TinyVLA(nn.Module):
    """Tiny VLA: CNN vision encoder + text embedding + fusion + action head.

    The vision encoder keeps a 1x2 spatial grid (left / right) via
    AdaptiveAvgPool2d((1, 2)) instead of collapsing to 1x1. A 1x1 pool would
    average the two halves and destroy the very positional signal the task
    needs ("which side is the target on?"). Keeping the left/right split lets
    the fusion head route the selected color to the correct action sign.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=16, action_dim=2):
        super().__init__()
        # Vision encoder: 32x32 -> 16 -> 8 -> 4, then 1x2 (keep left/right).
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),   # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 2)),               # 4x4 -> 1x2 (left/right)
            nn.Flatten(),                               # (B, 128)
        )
        # Language encoder
        self.text_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=VOCAB["<pad>"])
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim * 4, 32),
            nn.ReLU(),
        )
        # Fusion + action head (Tanh keeps actions in [-1, 1]).
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )

    def forward(self, images, texts):
        z_img = self.vision(images)               # (B, 128)
        z_txt = self.text_embed(texts)            # (B, 4, embed_dim)
        z_txt = z_txt.flatten(1)                  # (B, 4*embed_dim)
        z_txt = self.text_proj(z_txt)             # (B, 32)
        z = torch.cat([z_img, z_txt], dim=-1)     # (B, 160)
        action = self.fusion(z)                  # (B, 2)
        return action


# ============================================================
# 3. Training Loop (with train / val split)
# ============================================================

def train_toy_vla(epochs=100, lr=1e-3, batch_size=32, n_per_pattern=100,
                  device="cpu", seed=SEED):
    print("=" * 60)
    print("Toy VLA Training Demo -- Dual-Target (true multimodal fusion)")
    print("=" * 60)

    # Reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\nGenerating dual-target dataset ({n_per_pattern} samples/pattern)...")
    images, texts, actions, meta = generate_dataset(n_per_pattern=n_per_pattern,
                                                     seed=seed)
    n_patterns = len(COLOR_PAIRS) * 2
    print(f"  Base (image_layout, target_color) patterns: {n_patterns}")
    print(f"  Total samples: {len(images)}")
    print(f"  Image shape: {images[0].shape}")
    print(f"  Action shape: {actions[0].shape}")
    print(f"  Vocab size: {VOCAB_SIZE} -> {VOCAB}")

    splits = split_dataset(images, texts, actions, meta, seed=seed)
    print(f"  Split -> train: {len(splits['train']['images'])}, "
          f"val: {len(splits['val']['images'])}, "
          f"test: {len(splits['test']['images'])}")

    # Tensors per split.
    def to_tensors(split):
        return (
            torch.FloatTensor(split["images"]).to(device),
            torch.LongTensor(split["texts"]).to(device),
            torch.FloatTensor(split["actions"]).to(device),
        )

    tr_img, tr_txt, tr_act = to_tensors(splits["train"])
    va_img, va_txt, va_act = to_tensors(splits["val"])

    model = TinyVLA().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    print(f"\nTraining on {device} for {epochs} epochs...")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    train_losses, val_losses = [], []
    n_train = len(tr_img)
    for epoch in range(epochs):
        # --- train ---
        model.train()
        perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed + epoch))
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(tr_img[idx], tr_txt[idx])
            loss = mse(pred, tr_act[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # --- validate ---
        model.eval()
        with torch.no_grad():
            val_pred = model(va_img, va_txt)
            avg_val = mse(val_pred, va_act).item()
        val_losses.append(avg_val)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            marker = "***" if avg_train < 0.05 else ""
            print(f"  Epoch {epoch + 1:3d} | train: {avg_train:.4f} "
                  f"| val: {avg_val:.4f} {marker}")

    print(f"\nFinal train loss: {train_losses[-1]:.4f} | "
          f"val loss: {val_losses[-1]:.4f} "
          f"(started train {train_losses[0]:.4f})")
    return model, train_losses, val_losses, splits["test"]


# ============================================================
# 4. Evaluation & Visualization (with modality ablation)
# ============================================================

ACC_THRESHOLD = 0.5  # L2 error below this counts as a correct action.


def compute_accuracy(model, images_t, texts_t, actions_t, device="cpu"):
    """Return (accuracy in [0,1], predictions ndarray)."""
    model.eval()
    with torch.no_grad():
        preds = model(images_t.to(device), texts_t.to(device)).cpu().numpy()
    true = actions_t.cpu().numpy() if isinstance(actions_t, torch.Tensor) else actions_t
    errs = np.linalg.norm(preds - true, axis=1)
    acc = float((errs < ACC_THRESHOLD).mean())
    return acc, preds


def build_ablation_sets(test_split, device="cpu"):
    """Build the three evaluation tensors for the modality ablation.

    1. Full         : correct image + correct text
    2. Text-swapped : correct image + WRONG target text
    3. Image-flipped: image flipped left/right + correct text
    """
    images = test_split["images"]                       # (N, 3, 32, 32)
    texts = test_split["texts"]                         # (N, 4)
    actions = test_split["actions"]                     # (N, 2)
    meta = test_split["meta"]                           # list of (cL, cR, target)

    n = len(images)

    # Text-swapped: keep image, swap the text target to the OTHER color.
    shuf_texts = np.empty_like(texts)
    for i, (cL, cR, target) in enumerate(meta):
        other = cR if target == cL else cL
        shuf_texts[i] = tokenize_text(f"move to {other}")

    # Image-flipped: flip the image horizontally (swap target positions),
    # keep the correct text. A model that truly reads the image must now flip
    # its predicted action -> mismatch with the original true action.
    flip_images = np.ascontiguousarray(images[:, :, :, ::-1])

    return {
        "images": torch.FloatTensor(images).to(device),
        "texts": torch.LongTensor(texts).to(device),
        "actions": torch.FloatTensor(actions).to(device),
        "shuf_texts": torch.LongTensor(shuf_texts).to(device),
        "flip_images": torch.FloatTensor(flip_images).to(device),
    }


def run_modality_ablation(model, test_split, device="cpu"):
    """Run the three evaluation conditions and return a dict of results."""
    T = build_ablation_sets(test_split, device=device)
    full_acc, full_preds = compute_accuracy(
        model, T["images"], T["texts"], T["actions"], device)
    txt_acc, _ = compute_accuracy(
        model, T["images"], T["shuf_texts"], T["actions"], device)
    img_acc, _ = compute_accuracy(
        model, T["flip_images"], T["texts"], T["actions"], device)
    return {
        "full": (full_acc, full_preds),
        "text_swapped": (txt_acc, None),
        "image_flipped": (img_acc, None),
    }


def evaluate_and_visualize(model, train_losses, val_losses, test_split,
                           device="cpu"):
    print("\n" + "=" * 60)
    print("Evaluation: Predicted vs True Actions (test set)")
    print("=" * 60)

    T = build_ablation_sets(test_split, device=device)
    images_np = test_split["images"]
    actions_np = test_split["actions"]
    meta = test_split["meta"]
    full_acc, full_preds = compute_accuracy(
        model, T["images"], T["texts"], T["actions"], device)
    print(f"  Full-model test accuracy: {full_acc * 100:5.1f}%")

    # Pick 4 diverse test samples: one per target color if possible.
    chosen = []
    seen_targets = set()
    for i, (_, _, target) in enumerate(meta):
        if target not in seen_targets:
            chosen.append(i)
            seen_targets.add(target)
        if len(chosen) >= 4:
            break
    # Fallback if fewer than 4 distinct targets found.
    i = 0
    while len(chosen) < 4 and i < len(meta):
        if i not in chosen:
            chosen.append(i)
        i += 1
    chosen = chosen[:4]

    out_dir = Path(__file__).parent.parent / "results" / "vla"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 4-panel figure: predicted vs true actions for test samples ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Toy VLA (Dual-Target): Predicted vs True Actions",
                 fontsize=14, fontweight="bold")
    for ax, idx in zip(axes.flat, chosen):
        cL, cR, target = meta[idx]
        true_act = actions_np[idx]
        pred = full_preds[idx]
        err = float(np.linalg.norm(pred - true_act))
        status = "OK" if err < ACC_THRESHOLD else "FAIL"
        ax.arrow(0, 0, true_act[0], true_act[1], head_width=0.08, color="green",
                 label="True", alpha=0.7, linewidth=2)
        ax.arrow(0, 0, pred[0], pred[1], head_width=0.08, color="blue",
                 label="Predicted", alpha=0.7, linewidth=2)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"image: {cL}(L) {cR}(R)\n'{target}' -> "
                     f"pred={pred.round(2)} true={true_act.round(2)} "
                     f"[{status}]", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        # Inset thumbnail of the actual 32x32 image (H, W, C).
        inset = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
        inset.imshow(np.transpose(images_np[idx], (1, 2, 0)))
        inset.set_xticks([]); inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_edgecolor("black")
    plt.tight_layout()
    plt.savefig(out_dir / "toy_training.png", dpi=150)
    print(f"\n[Saved] {out_dir / 'toy_training.png'}")

    # ---- training + validation loss curves ----
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(train_losses, color="#0d6efd", label="train loss")
    ax2.plot(val_losses, color="#dc3545", label="val loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Toy VLA (Dual-Target) Training & Validation Loss")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0.05, color="green", linestyle="--", alpha=0.5,
                label="target threshold")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "toy_training_loss.png", dpi=150)
    print(f"[Saved] {out_dir / 'toy_training_loss.png'}")

    # ---- modality ablation ----
    results = run_modality_ablation(model, test_split, device=device)
    full = results["full"][0]
    txt_s = results["text_swapped"][0]
    img_s = results["image_flipped"][0]

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    labels = ["Full model", "Text-swapped", "Image-flipped"]
    values = [full, txt_s, img_s]
    colors = ["#2e8540", "#b8860b", "#8b0000"]
    bars = ax3.bar(labels, [v * 100 for v in values], color=colors,
                   edgecolor="black", linewidth=0.8)
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_ylim(0, 105)
    ax3.set_title("Modality Ablation: only the full model should succeed")
    for bar, v in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2, v * 100 + 2,
                 f"{v * 100:.1f}%", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "toy_ablation.png", dpi=150)
    print(f"[Saved] {out_dir / 'toy_ablation.png'}")

    # ---- printed ablation summary ----
    print("\n=== Modality Ablation Results ===")
    print(f"Full model accuracy:   {full * 100:5.1f}%")
    print(f"Text-swapped accuracy: {txt_s * 100:5.1f}%  (should be low)")
    print(f"Image-flipped accuracy:{img_s * 100:5.1f}%  (should be low)")
    return results


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}\n")

    model, train_losses, val_losses, test_split = train_toy_vla(device=device)
    evaluate_and_visualize(model, train_losses, val_losses, test_split,
                           device=device)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("  This demo forces *true* multimodal fusion:")
    print("    - Image carries WHERE each of two colored targets is.")
    print("    - Language carries WHICH target to move to.")
    print("    - Left/right assignment is randomized, so no modality alone works.")
    print("  The modality ablation quantifies this: corrupting either the text")
    print("  or the image tanks accuracy, while the full model stays accurate.")
    print("\n  Next step: run vla_demo.py --mode synthetic for a real API demo")
    print("=" * 60)


if __name__ == "__main__":
    main()
