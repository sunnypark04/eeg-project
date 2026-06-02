from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg_project.dataset import (
    EEGMotorImageryDataset,
    discover_subjects,
    make_subject_splits,
    INDEX_TO_LABEL,
)
from eeg_project.models import EEGCNN1D


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(y.detach().cpu().numpy().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_targets, all_preds),
        "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict[str, Any]:
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(y.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_targets, all_preds),
        "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(all_targets, all_preds, labels=[0, 1]).tolist(),
        "targets": all_targets,
        "predictions": all_preds,
        "probabilities": all_probs,
    }


def plot_confusion_matrix(cm, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    labels = [INDEX_TO_LABEL[0], INDEX_TO_LABEL[1]]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_experiment(args, channel_set: str) -> dict:
    set_seed(args.seed)

    device = get_device()
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    subjects = discover_subjects(args.data_dir)

    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]

    train_subjects, val_subjects, test_subjects = make_subject_splits(
        subjects,
        seed=args.seed,
    )

    print("\n" + "=" * 80)
    print(f"Channel set: {channel_set}")
    print(f"Device: {device}")
    print(f"Subjects found: {len(subjects)}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects: {len(val_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    print("=" * 80)

    train_dataset = EEGMotorImageryDataset(
        data_dir=args.data_dir,
        subjects=train_subjects,
        channel_set=channel_set,
        tmin=args.tmin,
        tmax=args.tmax,
        normalize=True,
    )

    val_dataset = EEGMotorImageryDataset(
        data_dir=args.data_dir,
        subjects=val_subjects,
        channel_set=channel_set,
        tmin=args.tmin,
        tmax=args.tmax,
        normalize=True,
    )

    test_dataset = EEGMotorImageryDataset(
        data_dir=args.data_dir,
        subjects=test_subjects,
        channel_set=channel_set,
        tmin=args.tmin,
        tmax=args.tmax,
        normalize=True,
    )

    print("Train summary:", train_dataset.summary())
    print("Val summary:", val_dataset.summary())
    print("Test summary:", test_dataset.summary())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = EEGCNN1D(
        num_channels=train_dataset.num_channels,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_val_macro_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        history.append(row)

        print(
            f"train loss={row['train_loss']:.4f}, "
            f"acc={row['train_accuracy']:.4f}, "
            f"f1={row['train_macro_f1']:.4f}"
        )

        print(
            f"val   loss={row['val_loss']:.4f}, "
            f"acc={row['val_accuracy']:.4f}, "
            f"f1={row['val_macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": best_val_macro_f1,
                "num_channels": train_dataset.num_channels,
                "channel_set": channel_set,
                "channel_names": train_dataset.channel_names,
                "tmin": args.tmin,
                "tmax": args.tmax,
                "seed": args.seed,
            }

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    model_path = results_dir / f"eeg_{channel_set}_seed{args.seed}.pt"
    torch.save(best_state, model_path)

    confusion_path = figures_dir / f"confusion_{channel_set}_seed{args.seed}.png"
    plot_confusion_matrix(
        cm=np.array(test_metrics["confusion_matrix"]),
        title=f"Confusion Matrix: {channel_set}",
        output_path=confusion_path,
    )

    result = {
        "channel_set": channel_set,
        "num_channels": train_dataset.num_channels,
        "channel_names": train_dataset.channel_names,
        "label_map": INDEX_TO_LABEL,
        "model_path": str(model_path),
        "data_dir": str(args.data_dir),
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "args": vars(args),
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "test_targets": test_metrics["targets"],
        "test_predictions": test_metrics["predictions"],
        "test_probabilities": test_metrics["probabilities"],
        "confusion_matrix_plot": str(confusion_path),
    }

    result_path = results_dir / f"eeg_{channel_set}_seed{args.seed}.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nFinal test result")
    print(f"Channel set: {channel_set}")
    print(f"Channels: {train_dataset.num_channels}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Result JSON saved to: {result_path}")

    return result


def plot_channel_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(summary_df))
    width = 0.35

    ax.bar(x - width / 2, summary_df["test_accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, summary_df["test_macro_f1"], width, label="Macro F1")

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["channel_set"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Full vs Reduced Channel EEG Classification")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train full/reduced EEG motor imagery models."
    )

    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument(
        "--channel-sets",
        nargs="+",
        default=["full", "motor21", "central3"],
        choices=["full", "motor21", "central3"],
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--tmin", type=float, default=0.5)
    parser.add_argument("--tmax", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default="results")

    return parser.parse_args()


def main():
    args = parse_args()

    all_results = []

    for channel_set in args.channel_sets:
        result = run_experiment(args, channel_set)
        all_results.append(result)

    summary_rows = []

    for result in all_results:
        summary_rows.append(
            {
                "channel_set": result["channel_set"],
                "num_channels": result["num_channels"],
                "test_accuracy": result["test_accuracy"],
                "test_macro_f1": result["test_macro_f1"],
                "best_val_macro_f1": result["best_val_macro_f1"],
                "model_path": result["model_path"],
                "confusion_matrix_plot": result["confusion_matrix_plot"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    results_dir = Path(args.results_dir)
    summary_path = results_dir / f"channel_comparison_seed{args.seed}.csv"
    summary_df.to_csv(summary_path, index=False)

    comparison_plot_path = results_dir / "figures" / f"channel_comparison_seed{args.seed}.png"
    plot_channel_comparison(summary_df, comparison_plot_path)

    print("\nChannel comparison summary")
    print(summary_df)
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved comparison plot to: {comparison_plot_path}")


if __name__ == "__main__":
    main()