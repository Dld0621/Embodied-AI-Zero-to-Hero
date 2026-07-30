"""
Unified PushCube — Minimal Action-Chunking Policy (Language-Conditioned)
========================================================================
A simplified action-chunking policy that learns to predict a chunk of
T future actions from a short history of K observation frames and a
language instruction, using a Transformer encoder.

NOTE / IMPORTANT — THIS IS NOT FULL ACT
----------------------------------------
This file is intentionally renamed from "ACT" to "Minimal Action-Chunking
Policy" because the previous version was not a real ACT: it used
seq_len=1 (so self-attention did nothing), had no CVAE latent variable,
and had no temporal ensembling (it just popped actions off a queue).

A *full* ACT (Zhao et al., 2023, "Learning Fine-Grained Bimanual
Manipulation with Low-Cost Hardware") additionally requires:
  1. A Conditional VAE (CVAE) with a learned latent variable z that is
     sampled from a learned posterior q(z | o, a) and regularized with a
     KL-divergence term. The latent injects multi-modality / style.
  2. Multi-step observation tokens on BOTH the CVAE encoder (style) side
     and the policy (decoder) side, fed as separate tokens to the
     Transformer so self-attention can reason over time.
  3. Temporal ensembling that aggregates *overlapping* action chunks
     with exponential weighting during execution (newer chunks weighted
     more heavily), instead of executing a single chunk until empty.

This teaching implementation implements the action-chunking backbone
together with SIMPLIFIED versions of (2) multi-frame observation tokens
and (3) exponential temporal ensembling, but it OMITS the CVAE (1).
Hence the honest name "Minimal Action-Chunking Policy".

Input:  K frames of 128x128 RGB history + tokenized language instruction
Output: action chunk (T steps of 2-D arm movement)

This is a teaching implementation, not a production policy.
"""

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

from unified_pushcube_env import PushCubeEnv, expert_action, COLOR_NAMES, CUBE_COLORS


# ----------------------------------------------------------------------
# Vocabulary & tokenization (shared with VLA track)
# ----------------------------------------------------------------------
VOCAB = {
    "<pad>": 0, "push": 1, "the": 2, "red": 3, "green": 4,
    "cube": 5, "to": 6, "right": 7, "left": 8, "top": 9,
    "bottom": 10, "and": 11, "center": 12,
}
MAX_LEN = 10

for _name in COLOR_NAMES:
    assert _name in VOCAB, f"Vocab missing color word: {_name}"


def tokenize(text):
    """Lowercase, split, map to ids, pad/truncate to MAX_LEN."""
    words = text.lower().replace(".", "").split()
    toks = [VOCAB.get(w, 0) for w in words]
    toks = toks[:MAX_LEN] + [0] * (MAX_LEN - len(toks))
    return toks


def train_action_chunking(args):
    """Train a minimal action-chunking policy with multi-frame tokens,
    temporal positional encoding, language conditioning, and exponential
    temporal ensembling."""
    print("=" * 70)
    print(" Unified PushCube — Minimal Action-Chunking Policy (lang-conditioned)")
    print("=" * 70)

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        print("[Error] PyTorch required. Install: pip install torch")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    action_dim = 2
    hist_len = args.hist_len          # number of observation tokens (K frames)
    chunk_size = args.chunk_size     # predicted action horizon (T)
    hidden_dim = 64
    vocab_size = len(VOCAB)
    embed_dim = 16

    # ------------------------------------------------------------------
    # Minimal Action-Chunking Policy (language-conditioned)
    # ------------------------------------------------------------------
    class MinimalActionChunkingPolicy(nn.Module):
        """Vision encoder -> K observation tokens + 1 language token ->
        Transformer encoder (seq_len = K+1) -> action chunk head.

        Temporal positional encoding is added via a learned embedding so
        the Transformer can distinguish frame order.
        """

        def __init__(self, action_dim=action_dim, chunk_size=chunk_size,
                     hist_len=hist_len, hidden_dim=hidden_dim,
                     vocab_size=vocab_size, embed_dim=embed_dim):
            super().__init__()
            self.action_dim = action_dim
            self.chunk_size = chunk_size
            self.hist_len = hist_len

            # Vision encoder: 128x128 -> hidden_dim (per frame)
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 8, 5, stride=2, padding=2),    # 64x64
                nn.ReLU(),
                nn.Conv2d(8, 16, 5, stride=2, padding=2),   # 32x32
                nn.ReLU(),
                nn.Conv2d(16, 16, 5, stride=2, padding=2),  # 16x16
                nn.ReLU(),
                nn.Conv2d(16, 8, 5, stride=2, padding=2),   # 8x8
                nn.ReLU(),
            )
            self.vision_fc = nn.Linear(8 * 8 * 8, hidden_dim)

            # Language encoder: word embeddings averaged over the sentence
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lang_fc = nn.Linear(embed_dim, hidden_dim)

            # Temporal positional encoding (learned)
            # K frames + 1 language token = K+1 positions
            self.temporal_pos = nn.Embedding(hist_len + 1, hidden_dim)

            # Transformer encoder over the K+1 tokens.
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

            # Action chunk head: predicts T * action_dim values
            self.action_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim * chunk_size),
                nn.Tanh(),
            )

        def encode_frame(self, image):
            # image: (N, 3, 128, 128) -> (N, hidden_dim)
            x = self.cnn(image)
            x = x.reshape(x.size(0), -1)
            return self.vision_fc(x)

        def forward(self, images, text_tokens):
            # images: (B, K, 3, 128, 128)
            # text_tokens: (B, MAX_LEN)
            B, K, C, H, W = images.shape

            # Encode every frame independently -> one token per frame.
            imgs_flat = images.reshape(B * K, C, H, W)
            vis_feats = self.encode_frame(imgs_flat)     # (B*K, hidden_dim)
            vis_feats = vis_feats.reshape(B, K, -1)      # (B, K, hidden_dim)

            # Encode language -> 1 token
            w = self.word_embed(text_tokens).mean(dim=1)  # (B, embed_dim)
            lang_feat = self.lang_fc(w)                   # (B, hidden_dim)
            lang_feat = lang_feat.unsqueeze(1)            # (B, 1, hidden_dim)

            # Concatenate: [frame_0, frame_1, ..., frame_{K-1}, language]
            all_tokens = torch.cat([vis_feats, lang_feat], dim=1)  # (B, K+1, hidden_dim)

            # Add temporal positional encoding
            positions = torch.arange(K + 1, device=all_tokens.device)
            pos_emb = self.temporal_pos(positions)       # (K+1, hidden_dim)
            all_tokens = all_tokens + pos_emb.unsqueeze(0)

            # Self-attention across the K+1 tokens.
            encoded = self.encoder(all_tokens)            # (B, K+1, hidden_dim)

            # Use the last token (language) for prediction — it has
            # attended to all visual frames and can integrate the goal.
            actions = self.action_head(encoded[:, -1])    # (B, T*action_dim)
            return actions.reshape(-1, self.chunk_size, self.action_dim)

    policy = MinimalActionChunkingPolicy(
        action_dim=action_dim,
        chunk_size=chunk_size,
        hist_len=hist_len,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # ------------------------------------------------------------------
    # Data Collection (uses the shared expert_action heuristic)
    # ------------------------------------------------------------------
    def collect_episodes(n_episodes):
        """Collect (history-of-K-frames, language, action-chunk-of-T) pairs
        using the shared expert_action(env) heuristic for demonstrations."""
        data_hist = []
        data_tokens = []
        data_chunks = []

        for ep in range(n_episodes):
            env = PushCubeEnv()
            obs = env.reset(seed=ep)

            # Language instruction for this episode
            lang = env.get_language_instruction()
            tok = tokenize(lang)

            images = []
            actions = []

            for _ in range(env.max_steps):
                img = env.render(size=128).transpose(2, 0, 1)  # (3,128,128)
                images.append(img)

                # Use the shared expert policy for demonstration actions.
                action = expert_action(env)
                actions.append(action)

                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break

            images = np.stack(images).astype(np.float32)   # (T, 3, 128, 128)
            actions = np.stack(actions).astype(np.float32)  # (T, action_dim)

            # Sliding window: every pred_interval steps, build a chunk of
            # chunk_size future actions and the K-frame observation history
            # ending at (and including) the current frame.
            step = args.pred_interval
            for i in range(0, len(actions) - chunk_size, step):
                chunk = actions[i:i + chunk_size]           # (T, action_dim)

                # History: the K frames up to and including frame i.
                start = max(0, i - hist_len + 1)
                hist = images[start:i + 1]                  # (<=K, 3, 128, 128)
                # Left-pad with the earliest available frame if needed so
                # that every sample has exactly hist_len tokens.
                if len(hist) < hist_len:
                    pad = np.repeat(hist[:1], hist_len - len(hist), axis=0)
                    hist = np.concatenate([pad, hist], axis=0)

                data_hist.append(hist)
                data_tokens.append(tok)
                data_chunks.append(chunk)

            if (ep + 1) % 20 == 0:
                print(f"  Collected {ep+1}/{n_episodes} episodes, "
                      f"{len(data_hist)} (history, chunk) pairs")

        return data_hist, data_tokens, data_chunks

    print(f"\nCollecting {args.n_episodes} demonstration episodes "
          f"(expert_action)...")
    hists, toks, chunks = collect_episodes(args.n_episodes)
    print(f"  Total chunks: {len(chunks)}")

    hists_t = torch.tensor(np.stack(hists), dtype=torch.float32).to(device)
    toks_t = torch.tensor(np.array(toks), dtype=torch.long).to(device)
    chunks_t = torch.tensor(np.stack(chunks), dtype=torch.float32).to(device)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nTraining for {args.epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        perm = torch.randperm(len(hists_t))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(hists_t), args.batch_size):
            idx = perm[i:i + args.batch_size]
            pred = policy(hists_t[idx], toks_t[idx])       # (B, T, action_dim)
            loss = F.mse_loss(pred, chunks_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")

    # ------------------------------------------------------------------
    # Evaluate with exponential temporal ensembling
    # ------------------------------------------------------------------
    n_eval = args.n_eval
    print(f"\nEvaluating with temporal ensembling "
          f"(deterministic, {n_eval} episodes)...")

    policy.eval()
    success_count = 0
    step_total = 0

    # Fixed eval seed for reproducibility
    torch.manual_seed(args.eval_seed)

    with torch.no_grad():
        for ep in range(n_eval):
            env = PushCubeEnv()
            obs = env.reset(seed=5000 + ep)

            # Language instruction for this episode
            lang = env.get_language_instruction()
            tok = tokenize(lang)
            tok_t = torch.tensor([tok], dtype=torch.long).to(device)

            # Frame history buffer (most-recent-last).
            frame_hist = deque(maxlen=hist_len)

            # Active predicted chunks: list of (chunk_array, start_step),
            # ordered oldest -> newest. We keep ALL chunks whose window
            # still covers the current step and average them.
            chunks_active = []  # list of (np.ndarray (T,action_dim), int start)

            done = False
            for step in range(env.max_steps):
                # Render current frame and append to history.
                img = env.render(size=128).transpose(2, 0, 1)
                frame_hist.append(img)
                # Left-pad history if we have not yet seen K frames.
                while len(frame_hist) < hist_len:
                    frame_hist.appendleft(img)

                # Predict a new chunk every pred_interval steps (and on the
                # very first step).
                if step % args.pred_interval == 0 or not chunks_active:
                    hist_arr = np.stack(list(frame_hist)).astype(np.float32)
                    hist_t = torch.tensor(hist_arr).unsqueeze(0).to(device)
                    chunk = policy(hist_t, tok_t).cpu().numpy()[0]  # (T, action_dim)
                    chunks_active.append((chunk, step))

                # Drop chunks whose window no longer covers this step.
                chunks_active = [
                    (c, s) for (c, s) in chunks_active
                    if step - s < len(c)
                ]

                # Exponential temporal ensembling over overlapping chunks.
                # Order newest -> oldest; weight newest most heavily:
                #   w_i = exp(-m * i),  i = 0 (newest) .. N-1 (oldest)
                ordered = list(reversed(chunks_active))  # newest first
                m = args.ensemble_weight
                num = np.zeros(action_dim, dtype=np.float32)
                den = 0.0
                for i, (c, s) in enumerate(ordered):
                    idx = step - s
                    if 0 <= idx < len(c):
                        w = math.exp(-m * i)
                        num += w * c[idx]
                        den += w
                action = num / (den + 1e-8)
                action = np.clip(action, -1.0, 1.0)

                obs, _, terminated, truncated, info = env.step(action)
                step_total += 1
                if terminated:
                    success_count += 1
                    done = True
                    break
                if truncated:
                    break

            status = "SUCCESS" if (done and info.get("is_success")) else "fail"
            print(f"  Eval {ep+1}/{n_eval}: {status}")

    success_rate = success_count / n_eval * 100 if n_eval > 0 else 0.0
    print(f"Success rate: {success_count}/{n_eval} = {success_rate:.1f}%")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), save_dir / "pushcube_action_chunking.pt")
    print(f"Model saved to {save_dir / 'pushcube_action_chunking.pt'}")

    results = {
        "task": "PushCube Minimal Action-Chunking Policy (lang-conditioned)",
        "note": ("Simplified action-chunking policy, NOT a full ACT "
                 "(no CVAE). Implements multi-frame observation tokens, "
                 "temporal positional encoding, language conditioning, "
                 "and exponential temporal ensembling."),
        "method": "Minimal Action-Chunking Policy",
        "has_cvae": False,
        "has_multi_step_obs_tokens": True,
        "has_temporal_pos_encoding": True,
        "has_language_conditioning": True,
        "has_temporal_ensembling": True,
        "n_episodes": args.n_episodes,
        "hist_len": args.hist_len,
        "chunk_size": args.chunk_size,
        "pred_interval": args.pred_interval,
        "ensemble_weight": args.ensemble_weight,
        "epochs": args.epochs,
        "best_loss": round(best_loss, 4),
        "n_eval": n_eval,
        "eval_seed": args.eval_seed,
        "success_rate": round(success_rate, 1),
        "success_count": success_count,
        "smoke_test": args.smoke_test,
    }
    with open(save_dir / "action_chunking_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_dir / 'action_chunking_results.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified PushCube — Minimal Action-Chunking Policy"
    )
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Demo episodes")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--chunk-size", type=int, default=10,
                        help="Action chunk size (horizon T)")
    parser.add_argument("--hist-len", type=int, default=3,
                        help="Number of observation tokens K (seq_len > 1)")
    parser.add_argument("--pred-interval", type=int, default=5,
                        help="Re-prediction interval (steps between chunks)")
    parser.add_argument("--ensemble-weight", type=float, default=0.01,
                        help="Exponential decay m for temporal ensembling "
                             "(w_i = exp(-m * i))")
    parser.add_argument("--eval-seed", type=int, default=1234,
                        help="Fixed RNG seed for deterministic evaluation")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="Number of eval episodes")
    parser.add_argument("--output-dir", type=str,
                        default="../results/unified_pushcube/action_chunking",
                        help="Output directory")
    parser.add_argument("--smoke-test", action="store_true",
                        help="CI smoke test: 2 episodes, 2 epochs, 2 eval")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_episodes = 2
        args.epochs = 2
        args.n_eval = 2
        print("[SMOKE TEST] n_episodes=2, epochs=2, n_eval=2")

    train_action_chunking(args)


if __name__ == "__main__":
    main()
