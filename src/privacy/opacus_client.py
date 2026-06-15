from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


IMG_SIZE = 224
BATCH_SIZE = 16

def freeze_feature_extractor(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

def disable_inplace_activations(module):
    for child in module.children():
        if hasattr(child, "inplace"):
            child.inplace = False
        disable_inplace_activations(child)

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
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return loader, len(ds)


def clean_state_dict(state_dict, reference_state):
    cleaned = {}

    for key, value in reference_state.items():
        cleaned[key] = value.detach().cpu().clone()

    for key, value in state_dict.items():
        clean_key = key.replace("_module.", "", 1)

        if clean_key in cleaned:
            cleaned[clean_key] = value.detach().cpu()

    return cleaned


def train_opacus_client(
    global_model,
    client_dir: Path,
    device,
    local_epochs=1,
    lr=1e-4,
    noise_multiplier=0.1,
    max_grad_norm=1.0,
    delta=1e-5,
):
    model = copy.deepcopy(global_model).to(device)
    disable_inplace_activations(model)
    freeze_feature_extractor(model)
    reference_state = global_model.state_dict()

    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)

    model.train()

    loader, num_samples = get_client_loader(client_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    privacy_engine = PrivacyEngine()

    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    total_loss = 0.0

    for _ in range(local_epochs):
        for images, labels in tqdm(loader, desc=f"Opacus DP training {client_dir.name}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    epsilon = privacy_engine.get_epsilon(delta)

    avg_loss = total_loss / max(1, len(loader) * local_epochs)
    state = clean_state_dict(model.state_dict(), reference_state)

    return state, num_samples, avg_loss, epsilon