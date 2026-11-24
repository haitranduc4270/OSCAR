"""Train the gene-embedding pathway-token model on the fixed BRCA split.

Selection uses 326 train and 82 validation samples. The epoch is the first
maximum of validation macro-F1. A new model is then trained for that many
epochs on the 408 development samples, and the 103 test samples are scored once.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from models.pathway_tokens import PathwayTokenModel, aggregate_pathway_tokens
from pathway_data import build_cohort, scale_matrices


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


def check_aggregate_example() -> None:
    embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    measurement = torch.tensor([[2.0, 4.0]])
    membership = torch.tensor([[1.0, 1.0]])
    token = aggregate_pathway_tokens(measurement, embedding, membership)
    if not torch.allclose(token, torch.tensor([[[1.0, 2.0]]])):
        raise AssertionError(f"Pathway mean example returned {token}")

    missing = torch.tensor([[2.0, float("nan")]])
    dropped = aggregate_pathway_tokens(missing, embedding, membership)
    if not torch.allclose(dropped, torch.tensor([[[2.0, 0.0]]])):
        raise AssertionError(f"Missing-gene mean returned {dropped}")


def make_loader(mrna, cnv, protein, y, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
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


def adamw(model: PathwayTokenModel, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, plain = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim <= 1 or name.endswith("bias"):
            plain.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def build_model(cfg: dict, cohort: dict) -> PathwayTokenModel:
    model_cfg = cfg["model"]
    return PathwayTokenModel(
        n_genes=len(cohort["genes"]),
        membership=cohort["membership"],
        total_index=cohort["total_index"],
        total_mask=cohort["total_mask"],
        phospho_index=cohort["phospho_index"],
        phospho_mask=cohort["phospho_mask"],
        key_index=cohort["key_index"],
        key_mask=cohort["key_mask"],
        num_classes=int(model_cfg["num_classes"]),
        gene_dim=int(model_cfg["gene_dim"]),
        token_dim=int(model_cfg["token_dim"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        dropout=float(model_cfg["dropout"]),
        use_gene_embedding=model_cfg.get("aggregation", "gene_vector") != "mean",
    )


def run_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    seen = 0
    for mrna, cnv, protein, y in loader:
        mrna = mrna.to(device)
        cnv = cnv.to(device)
        protein = protein.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(mrna, cnv, protein), y)
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
        mrna = mrna.to(device)
        cnv = cnv.to(device)
        protein = protein.to(device)
        y = y.to(device)
        batch_logits = model(mrna, cnv, protein)
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


def select_epoch(cfg: dict, cohort: dict, device: torch.device) -> tuple[int, list[dict]]:
    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    scaled = scale_matrices(
        cohort,
        cohort["train_idx"],
        {"train": cohort["train_idx"], "val": cohort["val_idx"]},
    )
    y = cohort["y"]
    train_loader = make_loader(
        scaled["mrna"]["train"],
        scaled["cnv"]["train"],
        scaled["protein"]["train"],
        y[cohort["train_idx"]],
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    val_loader = make_loader(
        scaled["mrna"]["val"],
        scaled["cnv"]["val"],
        scaled["protein"]["val"],
        y[cohort["val_idx"]],
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        seed=seed,
    )
    model = build_model(cfg, cohort).to(device)
    optimizer = adamw(model, float(cfg["train"]["lr"]), float(cfg["train"]["weight_decay"]))
    history = []
    best_epoch = 1
    best_f1 = -1.0
    max_epochs = int(cfg["train"]["max_epochs"])
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
        print(
            f"select epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
    return best_epoch, history


def retrain_and_test(cfg: dict, cohort: dict, epochs: int, device: torch.device) -> tuple[dict, dict]:
    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    retrain_idx = np.sort(np.concatenate([cohort["train_idx"], cohort["val_idx"]]))
    scaled = scale_matrices(
        cohort,
        retrain_idx,
        {"train": retrain_idx, "test": cohort["test_idx"]},
    )
    y = cohort["y"]
    train_loader = make_loader(
        scaled["mrna"]["train"],
        scaled["cnv"]["train"],
        scaled["protein"]["train"],
        y[retrain_idx],
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    test_loader = make_loader(
        scaled["mrna"]["test"],
        scaled["cnv"]["test"],
        scaled["protein"]["test"],
        y[cohort["test_idx"]],
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        seed=seed,
    )
    model = build_model(cfg, cohort).to(device)
    optimizer = adamw(model, float(cfg["train"]["lr"]), float(cfg["train"]["weight_decay"]))
    train_losses = []
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        train_losses.append({"epoch": epoch, "train_loss": train_loss})
        if epoch == epochs or epoch % 10 == 0:
            print(f"retrain epoch {epoch}/{epochs}: train_loss={train_loss:.4f}", flush=True)
    test_metrics = evaluate(model, test_loader, device)
    return test_metrics, {"state_dict": model.state_dict(), "train_losses": train_losses}


def main(config_path: str, dry_run: bool) -> None:
    check_aggregate_example()
    cfg = yaml.safe_load(open(config_path))
    configure_runtime()
    cohort = build_cohort(cfg)
    counts = cohort["split_counts"]
    expected = {"n_total": 511, "n_train": 326, "n_val": 82, "n_test": 103}
    if any(counts[key] != value for key, value in expected.items()):
        raise SystemExit(f"Split is {counts}, expected {expected}")
    print(
        f"genes {len(cohort['genes'])} of hallmark union {cohort['hallmark_union']}; "
        f"excluded {len(cohort['excluded_genes'])}; "
        f"split train/val/test {counts['n_train']}/{counts['n_val']}/{counts['n_test']}",
        flush=True,
    )
    print("program sizes", cohort["program_sizes"], flush=True)
    print("label counts", {key: counts[key] for key in ("train_labels", "val_labels", "test_labels")}, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dry_run:
        set_seed(int(cfg["train"]["seed"]))
        model = build_model(cfg, cohort).to(device)
        batch = 4
        logits = model(
            torch.zeros(batch, len(cohort["genes"]), device=device),
            torch.zeros(batch, len(cohort["genes"]), device=device),
            torch.zeros(batch, cohort["protein"].shape[1], device=device),
        )
        n_params = sum(parameter.numel() for parameter in model.parameters())
        print(f"dry run logits {tuple(logits.shape)} parameters {n_params}", flush=True)
        return

    chosen_epoch, history = select_epoch(cfg, cohort, device)
    test_metrics, trained = retrain_and_test(cfg, cohort, chosen_epoch, device)
    reference = float(cfg["scoring"]["reference_test_macro_f1"])

    log_cfg = cfg["logging"]
    out_dir = Path(log_cfg["log_dir"]) / log_cfg["log_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": config_path,
        "protocol": {
            "seed": int(cfg["data"]["seed"]),
            "n_train": counts["n_train"],
            "n_val": counts["n_val"],
            "n_retrain": counts["n_train"] + counts["n_val"],
            "n_test": counts["n_test"],
            "chosen_epoch": chosen_epoch,
            "selection": "first maximum of validation macro-F1",
        },
        "genes": {
            "hallmark_gmt": cfg["data"]["hallmark_gmt"],
            "hallmark_union": cohort["hallmark_union"],
            "embedding_genes": len(cohort["genes"]),
            "excluded_genes": cohort["excluded_genes"],
        },
        "programs": {
            "sizes": cohort["program_sizes"],
            "reads": {name: list(reads) for name, reads in zip(cohort["program_names"], cohort["program_reads"])},
        },
        "split": {
            "train_idx": cohort["train_idx"].tolist(),
            "val_idx": cohort["val_idx"].tolist(),
            "test_idx": cohort["test_idx"].tolist(),
            "train_labels": counts["train_labels"],
            "val_labels": counts["val_labels"],
            "test_labels": counts["test_labels"],
        },
        "selection_history": history,
        "retrain_history": trained["train_losses"],
        "test": test_metrics,
        "reference_test_macro_f1": reference,
        "beats_reference": bool(test_metrics["f1_macro"] > reference),
    }
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    torch.save(
        {
            "state_dict": trained["state_dict"],
            "chosen_epoch": chosen_epoch,
            "config": cfg,
        },
        out_dir / "retrain.pt",
    )
    print(f"Saved report to {report_path}", flush=True)
    print(
        f"test macro-F1 {test_metrics['f1_macro']:.4f} "
        f"accuracy {test_metrics['accuracy']:.4f} "
        f"per-class F1 {['%.4f' % value for value in test_metrics['f1_per_class']]} "
        f"support {test_metrics['support']}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_pathway_tokens_brca.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.config, args.dry_run)
