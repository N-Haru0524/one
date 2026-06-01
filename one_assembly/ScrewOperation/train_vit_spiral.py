import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from one_assembly.ScrewOperation.config import ScrewConfig, load_config, save_config, merge_cli_args
from one_assembly.ScrewOperation.model_builder import build_model
from one_assembly.ScrewOperation.dataset import SpiralDataset
from one_assembly.ScrewOperation.utils import make_mode_dir

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_dir(d: str) -> str:
    return os.path.join(BASE_DIR, d) if not os.path.isabs(d) else d


def resolve_train_paths(config: ScrewConfig):
    csv_paths = []
    image_dirs = []
    for d in config.train_dirs:
        d = _resolve_dir(d)
        csv = os.path.join(d, "samples.csv")
        imgs = os.path.join(d, "images")
        if not os.path.exists(csv):
            csv = os.path.join(d, "train", "samples.csv")
            imgs = os.path.join(d, "train", "images")
        csv_paths.append(csv)
        image_dirs.append(imgs)
    return csv_paths, image_dirs


def resolve_val_paths(train_dir: str):
    d = _resolve_dir(train_dir)
    val_csv = os.path.join(d, "validate_clear", "samples.csv")
    val_imgs = os.path.join(d, "validate_clear", "images")
    if not os.path.exists(val_csv):
        val_csv = os.path.join(d, "samples.csv")
        val_imgs = os.path.join(d, "images")
    return val_csv, val_imgs


def train(out_path: str, config: ScrewConfig, train_csv, train_image_dir, val_csv, val_image_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)
    print("num_classes:", config.num_classes)

    train_ds = SpiralDataset(train_csv, train_image_dir, config=config)
    val_ds = SpiralDataset(val_csv, val_image_dir, config=config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    model = build_model(config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    best_acc = 0.0
    best_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        acc = correct / max(total, 1)

        print(f"[{epoch:03d}] loss={avg_loss:.4f} val_acc={acc:.4f}")

        if acc > best_acc or (acc == best_acc and avg_loss < best_loss):
            best_acc = acc
            best_loss = avg_loss
            torch.save(model.state_dict(), out_path)
            print("  saved best model")

    print("best val acc:", best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--roi1", type=int, nargs=4, default=None)
    parser.add_argument("--roi2", type=int, nargs=4, default=None)

    args = parser.parse_args()

    first_config_path = os.path.join(
        BASE_DIR if not os.path.isabs(args.train_dirs[0]) else "",
        args.train_dirs[0], "config.yaml",
    )
    config = load_config(first_config_path)
    config = config.model_copy(update={"train_dirs": args.train_dirs})
    config = merge_cli_args(config, args)

    train_csv, train_image_dir = resolve_train_paths(config)
    print("train sources:", config.train_dirs)

    val_dir = args.val_dir or config.train_dirs[0]
    val_csv, val_image_dir = resolve_val_paths(val_dir)

    # Flat layout: datasets/model/{NNN}/. sequence / mode live in config.yaml
    # only (consistent with gen_pose_csv.py / data_collector.py).
    model_dir = make_mode_dir(BASE_DIR, "model")
    out_path = os.path.join(model_dir, "model.pt")
    save_config(config, os.path.join(model_dir, "config.yaml"))
    print("model dir:", model_dir)

    train(out_path, config, train_csv, train_image_dir, val_csv, val_image_dir)
