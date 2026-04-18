"""
Behavioral Cloning — train a policy by supervised imitation of TAS data.

Maps 31-dim state observations to action indices via cross-entropy loss.
Uses action-weighted loss to correct for heavy class imbalance (48% no-ops).
Applies Gaussian noise to kinematic features during training to reduce overfitting.

Usage:
    python src/train_bc.py
    python src/train_bc.py --data data/tas_transitions.pkl --epochs 200 --device cuda
    python src/train_bc.py --eval-only --model models/bc_best.pt
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from src.environment import CelesteEnv
from src.network import DuelingDQN


# Kinematic feature indices in the 31-dim state vector:
# [x/128, y/128, spd_x, spd_y, grace, djump, *tile_grid_25]
KINEMATIC_INDICES = [0, 1, 2, 3]  # x, y, spd_x, spd_y — add noise here only


class TASDataset(Dataset):
    def __init__(self, transitions, noise_std=0.02, augment=True):
        self.states = np.array([s for s, _ in transitions], dtype=np.float32)
        self.actions = np.array([a for _, a in transitions], dtype=np.int64)
        self.noise_std = noise_std
        self.augment = augment

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        state = self.states[idx].copy()
        if self.augment and self.noise_std > 0:
            state[KINEMATIC_INDICES] += np.random.normal(0, self.noise_std, len(KINEMATIC_INDICES)).astype(np.float32)
        return torch.FloatTensor(state), self.actions[idx]


def compute_class_weights(actions, n_actions):
    """Inverse-frequency weights to counteract action imbalance."""
    counts = np.bincount(actions, minlength=n_actions).astype(np.float32)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_actions  # normalize so mean weight = 1
    return torch.FloatTensor(weights)


def train(args):
    # Load data
    with open(args.data, 'rb') as f:
        transitions = pickle.load(f)

    print(f"Loaded {len(transitions)} transitions from {args.data}")

    n_actions = len(CelesteEnv.SIMPLE_ACTIONS)
    actions_all = np.array([a for _, a in transitions])
    class_weights = compute_class_weights(actions_all, n_actions)

    print(f"Class weights (to counteract imbalance):")
    for i, w in enumerate(class_weights):
        btn = CelesteEnv.SIMPLE_ACTIONS[i]
        count = int((actions_all == i).sum())
        print(f"  [{i:2d}] btn={btn:2d}  count={count:4d}  weight={w:.2f}")

    # Train / val split
    dataset = TASDataset(transitions, noise_std=args.noise_std, augment=True)
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    val_set.dataset = TASDataset(transitions, noise_std=0.0, augment=False)  # no noise on val

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    print(f"\nTrain: {train_size}  Val: {val_size}")

    # Model
    state_dim = transitions[0][0].shape[0]
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = DuelingDQN(state_dim, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    best_val_acc = 0.0
    best_val_loss = float('inf')

    print("=" * 60)
    print(f"TRAINING BC  |  epochs={args.epochs}  lr={args.lr}  noise={args.noise_std}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0

        for states, actions in train_loader:
            states, actions = states.to(device), actions.to(device)
            optimizer.zero_grad()
            logits = model(states)
            loss = criterion(logits, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            train_loss += loss.item() * len(actions)
            train_correct += (logits.argmax(1) == actions).sum().item()

        scheduler.step()

        train_loss /= train_size
        train_acc = train_correct / train_size

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for states, actions in val_loader:
                states, actions = states.to(device), actions.to(device)
                logits = model(states)
                val_loss += criterion(logits, actions).item() * len(actions)
                val_correct += (logits.argmax(1) == actions).sum().item()

        val_loss /= val_size
        val_acc = val_correct / val_size

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / f'{args.run_id}_best.pt')

        if epoch % args.log_interval == 0:
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train loss {train_loss:.4f}  acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f}  acc {val_acc:.3f} | "
                f"best val acc {best_val_acc:.3f}"
            )

    torch.save(model.state_dict(), save_dir / f'{args.run_id}_final.pt')
    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Saved {args.run_id}_best.pt and {args.run_id}_final.pt to {save_dir}")


def evaluate(args):
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    state_dim = CelesteEnv()._get_obs_dim()
    n_actions = len(CelesteEnv.SIMPLE_ACTIONS)

    model = DuelingDQN(state_dim, n_actions).to(device)
    model_path = args.model or f'models/{args.run_id}_best.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded {model_path}")

    env = CelesteEnv(room=args.room, max_steps=args.max_steps)

    completions = 0
    heights = []

    print("=" * 60)
    print(f"EVALUATION  |  {args.eval_episodes} episodes  room={args.room}")
    print("=" * 60)

    for ep in range(args.eval_episodes):
        state, _ = env.reset()
        while True:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = model(s).argmax(1).item()
            state, _, done, trunc, info = env.step(action)
            if done or trunc:
                break

        heights.append(info['max_height'])
        if info.get('completed', False):
            completions += 1
            print(f"  Ep {ep+1:>3}: COMPLETE")
        else:
            print(f"  Ep {ep+1:>3}: height={info['max_height']:.0f}")

    print("-" * 60)
    print(f"Success rate: {completions}/{args.eval_episodes} ({100*completions/args.eval_episodes:.0f}%)")
    print(f"Mean height:  {np.mean(heights):.1f}")
    print(f"Best height:  {min(heights):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Celeste BC Training')
    parser.add_argument('--data', type=str, default='data/tas_transitions.pkl')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--noise-std', type=float, default=0.02)
    parser.add_argument('--log-interval', type=int, default=25)
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--room', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--run-id', type=str, default='bc', help='Prefix for saved model files — use unique ID per run')
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args)
    else:
        train(args)
        evaluate(args)


if __name__ == '__main__':
    main()
