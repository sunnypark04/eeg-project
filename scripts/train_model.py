import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from my_dl_project.data.defungi import DEFUNGI_CLASSES, DeFungiDataset
from my_dl_project.models.simple_cnn import SimpleCNN
from my_dl_project.training.trainer import evaluate, train_one_epoch


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train a tiny CNN on the DeFungi dataset.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = int(config.get("image_size", 64))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = DeFungiDataset(config["data_dir"],transform=transform)
    num_samples = len(dataset)
    num_classes = len(DEFUNGI_CLASSES)

    val_fraction = float(config.get("val_fraction", 0.2))
    val_size = max(1, int(num_samples * val_fraction))
    train_size = num_samples - val_size
    if train_size < 1:
        raise RuntimeError("Dataset is too small to create a train/validation split.")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = SimpleCNN(num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.001)))

    epochs = int(config.get("epochs", 2))
    final_train_metrics = None
    final_val_metrics = None

    for epoch in range(1, epochs + 1):
        final_train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        final_val_metrics = evaluate(model, val_loader, loss_fn, device)

        print(
            f"epoch={epoch} "
            f"train_loss={final_train_metrics['loss']:.6f} "
            f"train_accuracy={final_train_metrics['accuracy']:.6f} "
            f"val_loss={final_val_metrics['loss']:.6f} "
            f"val_accuracy={final_val_metrics['accuracy']:.6f}",
            flush=True,
        )

    model_file = Path(config.get("model_file", "outputs/debug_model.pt"))
    metrics_file = Path(config.get("metrics_file", "outputs/train_debug_metrics.txt"))
    model_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": DEFUNGI_CLASSES,
            "config": config,
        },
        model_file,
    )

    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(f"num_classes={num_classes}\n")
        f.write(f"num_samples={num_samples}\n")
        f.write(f"final_train_loss={final_train_metrics['loss']:.6f}\n")
        f.write(f"final_train_accuracy={final_train_metrics['accuracy']:.6f}\n")
        f.write(f"final_val_loss={final_val_metrics['loss']:.6f}\n")
        f.write(f"final_val_accuracy={final_val_metrics['accuracy']:.6f}\n")
        f.write(f"model_file={model_file}\n")

    print("TRAINING_SUCCESS=true", flush=True)
    print(f"NUM_CLASSES={num_classes}", flush=True)
    print(f"NUM_SAMPLES={num_samples}", flush=True)
    print(f"METRICS_FILE={metrics_file}", flush=True)
    print(f"MODEL_FILE={model_file}", flush=True)
    print(f"FINAL_TRAIN_LOSS={final_train_metrics['loss']:.6f}", flush=True)
    print(f"FINAL_VAL_LOSS={final_val_metrics['loss']:.6f}", flush=True)


if __name__ == "__main__":
    main()
