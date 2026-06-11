from pathlib import Path
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


IMG_SIZE = 224
BATCH_SIZE = 16


def get_client_loader(client_dir: Path):
    tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    ds = datasets.ImageFolder(client_dir, transform=tfms)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    return loader, len(ds)


def train_client(global_model, client_dir: Path, device, local_epochs=1, lr=1e-4):
    model = copy.deepcopy(global_model).to(device)
    model.train()

    loader, num_samples = get_client_loader(client_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_loss = 0

    for _ in range(local_epochs):
        for images, labels in tqdm(loader, desc=f"Training {client_dir.name}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    avg_loss = total_loss / max(1, len(loader) * local_epochs)

    return model.state_dict(), num_samples, avg_loss