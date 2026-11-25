"""Shared data, runtime, training, and evaluation utilities for TokenOSCAR."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_runtime() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def _read_split_matrix(fold_dir: Path, stem: str) -> pd.DataFrame:
    """Genes by samples, train columns followed by validation columns."""
    train = pd.read_csv(fold_dir / f"{stem}_train.csv", index_col=0)
    val = pd.read_csv(fold_dir / f"{stem}_val.csv", index_col=0)
    val = val.reindex(train.index)
    if val.isna().any().any():
        raise ValueError(f"Train and val features do not align for {stem}")
    return pd.concat([train, val], axis=1)


def read_pair(fold_dir: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(fold_dir / f"{stem}_train.csv", index_col=0)
    val = pd.read_csv(fold_dir / f"{stem}_val.csv", index_col=0)
    if not train.index.equals(val.index):
        raise ValueError(f"{stem} train and val gene order differ in {fold_dir.name}")
    return train, val


def inner_split(y_dev: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y_dev))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y_dev,
        random_state=seed,
    )
    return np.sort(train_idx), np.sort(val_idx)


def make_loader(
    mrna: np.ndarray,
    cnv: np.ndarray,
    protein: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(mrna)),
        torch.from_numpy(np.ascontiguousarray(cnv)),
        torch.from_numpy(np.ascontiguousarray(protein)),
        torch.from_numpy(np.ascontiguousarray(y.astype(np.int64))),
    )
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def run_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    seen = 0
    for mrna, cnv, protein, y in loader:
        batch = {
            "mRNA": mrna.to(device),
            "CNV": cnv.to(device),
            "Proteomics": protein.to(device),
        }
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch), y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * y.shape[0]
        seen += y.shape[0]
    return total / seen


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    logits, labels = [], []
    total = 0.0
    seen = 0
    for mrna, cnv, protein, y in loader:
        batch = {
            "mRNA": mrna.to(device),
            "CNV": cnv.to(device),
            "Proteomics": protein.to(device),
        }
        y = y.to(device)
        batch_logits = model(batch)
        loss = F.cross_entropy(batch_logits, y)
        total += float(loss.item()) * y.shape[0]
        seen += y.shape[0]
        logits.append(batch_logits.detach().cpu())
        labels.append(y.detach().cpu())
    all_logits = torch.cat(logits)
    y_true = torch.cat(labels).numpy()
    y_pred = all_logits.argmax(dim=1).numpy()
    n_cls = all_logits.shape[1]
    label_ids = list(range(n_cls))
    per_class = f1_score(y_true, y_pred, average=None, labels=label_ids)
    return {
        "loss": total / seen,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=label_ids)),
        "f1_per_class": [float(value) for value in per_class],
        "support": np.bincount(y_true, minlength=n_cls).tolist(),
    }


def summarize(folds: list[dict]) -> dict:
    f1 = np.array([row["test"]["f1_macro"] for row in folds])
    acc = np.array([row["test"]["accuracy"] for row in folds])
    per_class = np.array([row["test"]["f1_per_class"] for row in folds])
    return {
        "f1_macro_mean": float(f1.mean()),
        "f1_macro_std": float(f1.std()),
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std()),
        "f1_per_class_mean": per_class.mean(axis=0).tolist(),
    }


def run_oscar_fold(oscar_cfg, dims, fold, device, max_epochs: int, build) -> dict:
    seed = int(oscar_cfg["train"]["seed"])
    batch_size = int(oscar_cfg["train"]["batch_size"])
    lr = float(oscar_cfg["train"]["lr"])
    weight_decay = float(oscar_cfg["train"]["weight_decay"])
    y_dev, y_test = fold["y_dev"], fold["y_test"]
    train_idx, val_idx = inner_split(y_dev, seed)

    def transform_all(fit_idx, parts):
        scaled = {}
        for name in ("mRNA", "CNV", "Proteomics"):
            scaler = StandardScaler().fit(fold["dev"][name][fit_idx])
            scaled[name] = {}
            for part, idx in parts.items():
                source = fold["test"][name] if part == "test" else fold["dev"][name][idx]
                scaled[name][part] = scaler.transform(source).astype(np.float32)
        return scaled

    set_seed(seed)
    scaled = transform_all(train_idx, {"train": train_idx, "val": val_idx})
    train_loader = make_loader(
        scaled["mRNA"]["train"], scaled["CNV"]["train"], scaled["Proteomics"]["train"],
        y_dev[train_idx], batch_size, True, seed,
    )
    val_loader = make_loader(
        scaled["mRNA"]["val"], scaled["CNV"]["val"], scaled["Proteomics"]["val"],
        y_dev[val_idx], batch_size, False, seed,
    )
    model = build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_epoch, best_f1 = 1, -1.0
    for epoch in range(1, max_epochs + 1):
        run_epoch(model, train_loader, optimizer, device)
        score = evaluate(model, val_loader, device)["f1_macro"]
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch

    set_seed(seed)
    dev_idx = np.arange(len(y_dev))
    scaled = transform_all(dev_idx, {"train": dev_idx, "test": None})
    train_loader = make_loader(
        scaled["mRNA"]["train"], scaled["CNV"]["train"], scaled["Proteomics"]["train"],
        y_dev, batch_size, True, seed,
    )
    test_loader = make_loader(
        scaled["mRNA"]["test"], scaled["CNV"]["test"], scaled["Proteomics"]["test"],
        y_test, batch_size, False, seed,
    )
    model = build().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _epoch in range(best_epoch):
        run_epoch(model, train_loader, optimizer, device)
    test_metrics = evaluate(model, test_loader, device)
    return {
        "n_inner_train": int(len(train_idx)),
        "n_inner_val": int(len(val_idx)),
        "n_retrain": int(len(y_dev)),
        "n_test": int(len(y_test)),
        "chosen_epoch": best_epoch,
        "inner_val_f1_macro": best_f1,
        "test": test_metrics,
    }
