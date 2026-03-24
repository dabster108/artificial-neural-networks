import argparse
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


@dataclass
class Config:
    n_samples: int = 6000
    n_classes: int = 3
    noise: float = 0.25
    batch_size: int = 128
    hidden_dim: int = 128
    n_blocks: int = 4
    dropout: float = 0.15
    epochs: int = 80
    lr: float = 3e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    patience: int = 12
    label_smoothing: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: str = "best_spiral_model.pt"
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_spiral_dataset(
    n_samples: int,
    n_classes: int,
    noise: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    points_per_class = n_samples // n_classes

    features = []
    labels = []

    for class_idx in range(n_classes):
        r = torch.linspace(0.0, 1.0, points_per_class)
        t = (
            torch.linspace(class_idx * 4.0, (class_idx + 1) * 4.0, points_per_class)
            + torch.randn(points_per_class, generator=generator) * noise
        )
        x = r * torch.sin(t)
        y = r * torch.cos(t)

        class_points = torch.stack([x, y], dim=1)
        class_labels = torch.full((points_per_class,), class_idx, dtype=torch.long)

        features.append(class_points)
        labels.append(class_labels)

    x_all = torch.cat(features, dim=0)
    y_all = torch.cat(labels, dim=0)

    perm = torch.randperm(x_all.size(0), generator=generator)
    return x_all[perm], y_all[perm]


def standardize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std, mean, std


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class AdvancedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        n_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_norm(x)
        return self.head(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_samples += xb.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train(cfg: Config) -> None:
    set_seed(cfg.seed)

    x, y = make_spiral_dataset(
        n_samples=cfg.n_samples,
        n_classes=cfg.n_classes,
        noise=cfg.noise,
        seed=cfg.seed,
    )
    x, mean, std = standardize(x)

    dataset = TensorDataset(x, y)
    n_total = len(dataset)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = AdvancedMLP(
        input_dim=2,
        hidden_dim=cfg.hidden_dim,
        n_classes=cfg.n_classes,
        n_blocks=cfg.n_blocks,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.device == "cuda"))

    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(cfg.device == "cuda")):
                logits = model(xb)
                loss = F.cross_entropy(
                    logits,
                    yb,
                    label_smoothing=cfg.label_smoothing,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * xb.size(0)
            running_correct += (logits.argmax(dim=1) == yb).sum().item()
            running_total += xb.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate(model, val_loader, cfg.device)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "config": cfg.__dict__,
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                },
                cfg.checkpoint_path,
            )

        if epoch - best_epoch >= cfg.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, cfg.device)
    print("\nFinal Evaluation")
    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(f"Best val accuracy: {checkpoint['best_val_acc']:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Inference example on a single custom point.
    sample = torch.tensor([[0.2, -0.3]], dtype=torch.float32)
    sample = (sample - checkpoint["mean"]) / checkpoint["std"]
    with torch.no_grad():
        probs = F.softmax(model(sample.to(cfg.device)), dim=1).cpu().squeeze(0)

    pred_class = int(probs.argmax().item())
    print("\nInference Example")
    print("Input point: [0.2, -0.3]")
    print(f"Predicted class: {pred_class}")
    print(f"Class probabilities: {probs.tolist()}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Advanced PyTorch neural network from scratch")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        lr=args.lr,
        noise=args.noise,
        seed=args.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
