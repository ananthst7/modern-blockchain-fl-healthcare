from pathlib import Path
from collections import defaultdict
from models.efficientnet import build_efficientnet_b0

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def get_client_dataloader(client_dir, batch_size=16):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = datasets.ImageFolder(
        root=str(client_dir),
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    return loader, len(dataset)


def is_trainable_tensor_key(key):
    if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
        return False

    return key.endswith("weight") or key.endswith("bias")


def train_ila_client(
    global_model,
    client_dir,
    device,
    local_epochs=1,
    lr=1e-4,
    batch_size=16,
):
    """
    Local client training that also estimates:
    1. Fisher-like score: mean(gradient^2)
    2. Gradient variance score: variance of gradient norms across batches

    These scores are used by ILA-CKKS to identify tensors that are both
    update-important and potentially privacy-sensitive.
    """

    client_model = build_efficientnet_b0(
        num_classes=2,
        pretrained=True,
    )
    client_model.load_state_dict(global_model.state_dict())
    client_model.to(device)
    client_model.train()

    loader, dataset_size = get_client_dataloader(client_dir, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(client_model.parameters(), lr=lr)

    fisher_accumulator = defaultdict(float)
    grad_norm_values = defaultdict(list)

    total_loss = 0.0
    total_batches = 0

    named_params = {
        name: param
        for name, param in client_model.named_parameters()
        if param.requires_grad
    }

    for _ in range(local_epochs):
        progress = tqdm(loader, desc=f"ILA training {Path(client_dir).name}", leave=False)

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = client_model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            for name, param in named_params.items():
                if param.grad is None:
                    continue

                if not is_trainable_tensor_key(name):
                    continue

                grad = param.grad.detach()

                fisher_accumulator[name] += torch.mean(grad.pow(2)).item()
                grad_norm_values[name].append(torch.norm(grad).item())

            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)

    fisher_scores = {}
    gradvar_scores = {}

    for key, value in fisher_accumulator.items():
        fisher_scores[key] = value / max(total_batches, 1)

    for key, values in grad_norm_values.items():
        if len(values) <= 1:
            gradvar_scores[key] = 0.0
        else:
            grad_tensor = torch.tensor(values, dtype=torch.float32)
            gradvar_scores[key] = torch.var(grad_tensor, unbiased=False).item()

    state = {
        key: value.detach().cpu()
        for key, value in client_model.state_dict().items()
    }

    return state, dataset_size, avg_loss, fisher_scores, gradvar_scores