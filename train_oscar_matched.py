"""Run the original OSCAR model on the pathway-token seed-42 split.

Epoch selection matches the pathway-token run: first maximum of validation
macro-F1, then a new model trained for that many epochs on the 408
development samples, then one test evaluation. Features stay the full CNV,
mRNA and RPPA matrices. Hyperparameters stay those in the OSCAR BRCA config.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.self_attn_fusion import SelfAttentionFusionModel
from pathway_data import _read_split_matrix, build_cohort
from train_pathway_tokens import configure_runtime, set_seed


def load_full_matrices(cfg: dict) -> dict[str, np.ndarray]:
    fold_dir = Path(cfg["data"]["pre_split_fold"])
    frames = {
        "mRNA": _read_split_matrix(fold_dir, cfg["data"]["mrna_stem"]),
        "CNV": _read_split_matrix(fold_dir, cfg["data"]["cnv_stem"]),
        "Proteomics": _read_split_matrix(fold_dir, cfg["data"]["protein_stem"]),
    }
    columns = [list(frame.columns) for frame in frames.values()]
    if columns[0] != columns[1] or columns[0] != columns[2]:
        raise ValueError("Sample columns differ across omics")
    return {name: frame.T.to_numpy(dtype=np.float64) for name, frame in frames.items()}


def scale_full(matrices, fit_idx, apply_idx):
    scaled = {}
    for name, values in matrices.items():
        scaler = StandardScaler()
        scaler.fit(values[fit_idx])
        scaled[name] = {
            split: scaler.transform(values[idx]).astype(np.float32)
            for split, idx in apply_idx.items()
        }
    return scaled


def make_loader(mrna, cnv, protein, y, batch_size, shuffle, seed):
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


def build_oscar(oscar_cfg: dict, dims: dict[str, int]) -> SelfAttentionFusionModel:
    model_cfg = oscar_cfg["model"]
    train_cfg = oscar_cfg["train"]
    return SelfAttentionFusionModel(
        input_dims=dims,
        num_classes=int(model_cfg["num_classes"]),
        latent_dim=int(model_cfg["latent_dim"]),
        encoder_hidden_dim=int(model_cfg["hidden_dim"]),
        fusion_hidden=int(model_cfg["fusion_hidden"]),
        dropout=float(model_cfg["dropout"]),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        fusion_type=model_cfg["fusion_type"],
        use_cross_attention=bool(model_cfg["use_cross_attention"]),
        cross_attn_hidden=int(model_cfg["cross_attn_hidden"]),
        use_original_for_proteomics=bool(model_cfg["use_original_for_proteomics"]),
        use_gated_fusion=bool(model_cfg["use_gated_fusion"]),
        gated_fusion_hidden=int(model_cfg["gated_fusion_hidden"]),
    )


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
    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(logits).argmax(dim=1).numpy()
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3])
    return {
        "loss": total / seen,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3])),
        "f1_per_class": [float(value) for value in per_class],
        "support": np.bincount(y_true, minlength=4).tolist(),
    }


def main() -> None:
    pathway_cfg = yaml.safe_load(open("configs/config_pathway_tokens_brca.yaml"))
    oscar_cfg = yaml.safe_load(open("configs/config_self_attn_cross_attn_brca.yaml"))
    configure_runtime()
    cohort = build_cohort(pathway_cfg)
    counts = cohort["split_counts"]
    if (counts["n_train"], counts["n_val"], counts["n_test"]) != (326, 82, 103):
        raise SystemExit(f"Unexpected split {counts}")
    matrices = load_full_matrices(pathway_cfg)
    if matrices["mRNA"].shape[0] != counts["n_total"]:
        raise SystemExit("Full matrix sample count does not match the pathway split")
    dims = {name: values.shape[1] for name, values in matrices.items()}
    print(f"OSCAR input dims {dims}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(oscar_cfg["train"]["seed"])
    batch_size = int(oscar_cfg["train"]["batch_size"])
    max_epochs = int(pathway_cfg["train"]["max_epochs"])
    lr = float(oscar_cfg["train"]["lr"])
    weight_decay = float(oscar_cfg["train"]["weight_decay"])
    y = cohort["y"]

    set_seed(seed)
    scaled = scale_full(
        matrices,
        cohort["train_idx"],
        {"train": cohort["train_idx"], "val": cohort["val_idx"]},
    )
    train_loader = make_loader(
        scaled["mRNA"]["train"], scaled["CNV"]["train"], scaled["Proteomics"]["train"],
        y[cohort["train_idx"]], batch_size, True, seed,
    )
    val_loader = make_loader(
        scaled["mRNA"]["val"], scaled["CNV"]["val"], scaled["Proteomics"]["val"],
        y[cohort["val_idx"]], batch_size, False, seed,
    )
    model = build_oscar(oscar_cfg, dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    best_epoch = 1
    best_f1 = -1.0
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_f1_macro": val_metrics["f1_macro"],
        })
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
            print(
                f"select epoch {epoch}: val_f1_macro={val_metrics['f1_macro']:.4f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )

    set_seed(seed)
    retrain_idx = np.sort(np.concatenate([cohort["train_idx"], cohort["val_idx"]]))
    scaled = scale_full(matrices, retrain_idx, {"train": retrain_idx, "test": cohort["test_idx"]})
    train_loader = make_loader(
        scaled["mRNA"]["train"], scaled["CNV"]["train"], scaled["Proteomics"]["train"],
        y[retrain_idx], batch_size, True, seed,
    )
    test_loader = make_loader(
        scaled["mRNA"]["test"], scaled["CNV"]["test"], scaled["Proteomics"]["test"],
        y[cohort["test_idx"]], batch_size, False, seed,
    )
    model = build_oscar(oscar_cfg, dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(1, best_epoch + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        if epoch == best_epoch or epoch % 10 == 0:
            print(f"retrain epoch {epoch}/{best_epoch}: train_loss={train_loss:.4f}", flush=True)
    test_metrics = evaluate(model, test_loader, device)

    out_dir = Path("lightning_logs/oscar_matched_brca")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "self_attn_fusion",
        "config": "configs/config_self_attn_cross_attn_brca.yaml",
        "protocol": {
            "seed": int(pathway_cfg["data"]["seed"]),
            "oscar_init_seed": seed,
            "n_train": 326,
            "n_val": 82,
            "n_retrain": 408,
            "n_test": 103,
            "chosen_epoch": best_epoch,
            "max_epochs": max_epochs,
            "selection": "first maximum of validation macro-F1",
        },
        "input_dims": dims,
        "selection_history": history,
        "test": test_metrics,
    }
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved report to {report_path}", flush=True)
    print(
        f"test macro-F1 {test_metrics['f1_macro']:.4f} "
        f"accuracy {test_metrics['accuracy']:.4f} "
        f"per-class F1 {['%.4f' % value for value in test_metrics['f1_per_class']]} "
        f"support {test_metrics['support']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
