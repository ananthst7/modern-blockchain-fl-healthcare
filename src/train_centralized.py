from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

from models.efficientnet import build_efficientnet_b0


DATA_DIR = Path("data/processed/covid_binary")
RESULTS_DIR = Path("results/centralized")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
IMG_SIZE = 224
NUM_CLASSES = 2


def get_loaders():
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_ds.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "loss": total_loss / len(loader),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, test_loader, classes = get_loaders()
    print("Classes:", classes)

    model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    history = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

    total_time = time.time() - start_time

    final_results = {
        "model": "EfficientNet-B0",
        "dataset": "COVID Radiography Binary",
        "classes": classes,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "device": str(device),
        "total_training_time_sec": total_time,
        "history": history,
        "final_metrics": history[-1],
    }

    with open(RESULTS_DIR / "covid_efficientnet_centralized.json", "w") as f:
        json.dump(final_results, f, indent=4)

    torch.save(model.state_dict(), RESULTS_DIR / "covid_efficientnet_centralized.pth")

    print("\nSaved results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()