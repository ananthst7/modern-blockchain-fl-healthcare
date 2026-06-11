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


def state_l2_distance(model, global_state, device):
    reg = torch.tensor(0.0, device=device)

    for name, param in model.named_parameters():
        if name in global_state:
            global_param = global_state[name].to(device)
            reg += torch.sum((param - global_param) ** 2)

    return reg


def linear_correction(model, h_state, device):
    correction = torch.tensor(0.0, device=device)

    for name, param in model.named_parameters():
        if name in h_state:
            h = h_state[name].to(device)
            correction += torch.sum(param * h)

    return correction


def train_feddyn_client(
    global_model,
    client_dir: Path,
    device,
    h_state,
    alpha=0.01,
    local_epochs=1,
    lr=1e-4,
):
    model = copy.deepcopy(global_model).to(device)
    model.train()

    global_state = {
        k: v.detach().clone().to(device)
        for k, v in global_model.state_dict().items()
        if torch.is_floating_point(v)
    }

    loader, num_samples = get_client_loader(client_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_loss = 0.0

    for _ in range(local_epochs):
        for images, labels in tqdm(loader, desc=f"FedDyn training {client_dir.name}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            ce_loss = criterion(outputs, labels)

            reg_loss = (alpha / 2.0) * state_l2_distance(model, global_state, device)
            dyn_loss = linear_correction(model, h_state, device)

            loss = ce_loss + reg_loss - dyn_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    avg_loss = total_loss / max(1, len(loader) * local_epochs)

    client_state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
    }

    return client_state, num_samples, avg_loss