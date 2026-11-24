"""Five-fold comparison on the pre-split BRCA folds.

Each fold's validation file is the held-out test. Epochs are chosen on a
stratified 20% slice of that fold's training file, then a new model is trained
on the whole training file and scored once. The reported number is the mean
and standard deviation of those five test macro-F1 scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pathway_data import _read_split_matrix, build_cohort
from models.moxgate import MoXGATEModel
from models.moxgate_pathway import MoXGATEWithPathway
from models.oscar_pathway import OscarPathwayModel
from train_oscar_matched import build_oscar, evaluate as evaluate_oscar, run_epoch as run_oscar_epoch
from train_pathway_tokens import (
    adamw,
    build_model,
    configure_runtime,
    evaluate,
    make_loader,
    run_epoch,
    set_seed,
)

FOLDS = Path("csv/processed/BRCA/5-fold")


def read_pair(fold_dir: Path, stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(fold_dir / f"{stem}_train.csv", index_col=0)
    val = pd.read_csv(fold_dir / f"{stem}_val.csv", index_col=0)
    if not train.index.equals(val.index):
        raise ValueError(f"{stem} train and val gene order differ in {fold_dir.name}")
    return train, val


def load_fold(fold_dir: Path, reference_columns: dict[str, pd.Index]) -> dict:
    stems = {
        "mRNA": "BRCA_mRNA",
        "CNV": "BRCA_CNV_1_channel",
        "Proteomics": "BRCA_proteomics",
    }
    pairs = {name: read_pair(fold_dir, stem) for name, stem in stems.items()}
    columns = list(pairs["mRNA"][0].columns)
    for name, (train, _val) in pairs.items():
        if list(train.columns) != columns:
            raise ValueError(f"{name} sample order differs in {fold_dir.name}")
        if not train.index.equals(reference_columns[name]):
            raise ValueError(f"{name} feature order differs in {fold_dir.name}")
    y_dev = pd.read_csv(fold_dir / "BRCA_label_num_train.csv").iloc[:, 0].to_numpy(dtype=np.int64)
    y_test = pd.read_csv(fold_dir / "BRCA_label_num_val.csv").iloc[:, 0].to_numpy(dtype=np.int64)
    if len(y_dev) != len(columns) or len(y_test) != pairs["mRNA"][1].shape[1]:
        raise ValueError(f"Label count does not match samples in {fold_dir.name}")
    dev, test = {}, {}
    for name, (train, val) in pairs.items():
        dev[name] = train.T.to_numpy(dtype=np.float64)
        test[name] = val.T.to_numpy(dtype=np.float64)
    return {"dev": dev, "test": test, "y_dev": y_dev, "y_test": y_test}


def inner_split(y_dev: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y_dev))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y_dev,
        random_state=seed,
    )
    return np.sort(train_idx), np.sort(val_idx)


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


def run_oscar_fold(oscar_cfg, dims, fold, device, max_epochs: int, build=None) -> dict:
    seed = int(oscar_cfg["train"]["seed"])
    batch_size = int(oscar_cfg["train"]["batch_size"])
    lr = float(oscar_cfg["train"]["lr"])
    weight_decay = float(oscar_cfg["train"]["weight_decay"])
    y_dev, y_test = fold["y_dev"], fold["y_test"]
    train_idx, val_idx = inner_split(y_dev, seed)
    factory = build or (lambda: build_oscar(oscar_cfg, dims))

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
    model = factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_epoch, best_f1 = 1, -1.0
    for epoch in range(1, max_epochs + 1):
        run_oscar_epoch(model, train_loader, optimizer, device)
        score = evaluate_oscar(model, val_loader, device)["f1_macro"]
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
    model = factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _epoch in range(best_epoch):
        run_oscar_epoch(model, train_loader, optimizer, device)
    test_metrics = evaluate_oscar(model, test_loader, device)
    return {
        "n_inner_train": int(len(train_idx)),
        "n_inner_val": int(len(val_idx)),
        "n_retrain": int(len(y_dev)),
        "n_test": int(len(y_test)),
        "chosen_epoch": best_epoch,
        "inner_val_f1_macro": best_f1,
        "test": test_metrics,
    }


def build_moxgate(cfg: dict, dims: dict[str, int]) -> MoXGATEModel:
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    return MoXGATEModel(
        input_dims=dims,
        num_classes=int(model_cfg["num_classes"]),
        latent_dim=int(model_cfg["latent_dim"]),
        classifier_dim=int(model_cfg.get("classifier_dim", model_cfg.get("fusion_hidden", 128))),
        attn_dropout=float(model_cfg.get("attn_dropout", model_cfg.get("dropout", 0.1))),
        classifier_dropout=float(model_cfg.get("classifier_dropout", 0.3)),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        n_heads=int(model_cfg["n_heads"]),
        joint_n_heads=int(model_cfg["joint_n_heads"]),
        use_focal_loss=bool(model_cfg["use_focal_loss"]),
        focal_gamma=float(model_cfg["focal_gamma"]),
        lambda_modality=float(
            model_cfg.get("lambda_modality", model_cfg.get("modality_balance_weight", 1e-2))
        ),
        lambda_cross_frobenius=float(model_cfg.get("lambda_cross_frobenius", 1e-2)),
    )


def run_moxgate_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    seen = 0
    for mrna, cnv, protein, y in loader:
        batch = {"mRNA": mrna.to(device), "CNV": cnv.to(device), "Proteomics": protein.to(device)}
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model._compute_loss(model(batch), y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * y.shape[0]
        seen += y.shape[0]
    return total / seen


def run_moxgate_fold(cfg, dims, fold, device, max_epochs: int, build=None) -> dict:
    seed = int(cfg["train"]["seed"])
    batch_size = int(cfg["train"]["batch_size"])
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
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
    factory = build or (lambda: build_moxgate(cfg, dims))
    model = factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_epoch, best_f1 = 1, -1.0
    for _epoch in range(1, max_epochs + 1):
        run_moxgate_epoch(model, train_loader, optimizer, device)
        score = evaluate_oscar(model, val_loader, device)["f1_macro"]
        if score > best_f1:
            best_f1 = score
            best_epoch = _epoch

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
    model = factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _epoch in range(best_epoch):
        run_moxgate_epoch(model, train_loader, optimizer, device)
    return {
        "n_inner_train": int(len(train_idx)),
        "n_inner_val": int(len(val_idx)),
        "n_retrain": int(len(y_dev)),
        "n_test": int(len(y_test)),
        "chosen_epoch": best_epoch,
        "inner_val_f1_macro": best_f1,
        "test": evaluate_oscar(model, test_loader, device),
    }


def main() -> None:
    pathway_cfg = yaml.safe_load(open("configs/config_pathway_tokens_brca.yaml"))
    mean_cfg = yaml.safe_load(open("configs/config_pathway_mean_brca.yaml"))
    oscar_cfg = yaml.safe_load(open("configs/config_self_attn_cross_attn_brca.yaml"))
    moxgate_cfg = yaml.safe_load(open("configs/config_moxgate_brca.yaml"))
    configure_runtime()
    reference = _read_split_matrix(FOLDS / "fold_1", "BRCA_mRNA")
    reference_columns = {
        "mRNA": reference.index,
        "CNV": _read_split_matrix(FOLDS / "fold_1", "BRCA_CNV_1_channel").index,
        "Proteomics": _read_split_matrix(FOLDS / "fold_1", "BRCA_proteomics").index,
    }
    cohort = build_cohort(pathway_cfg)
    gene_positions = [reference_columns["mRNA"].get_loc(gene) for gene in cohort["genes"]]
    cnv_positions = [reference_columns["CNV"].get_loc(gene) for gene in cohort["genes"]]
    if gene_positions != cnv_positions and reference_columns["mRNA"].equals(reference_columns["CNV"]):
        raise SystemExit("CNV and mRNA positions diverged")
    # CNV has fewer rows than mRNA. Locate genes on each index separately.
    cohort["cnv_positions"] = [int(reference_columns["CNV"].get_loc(gene)) for gene in cohort["genes"]]
    cohort["gene_positions"] = [int(pos) for pos in gene_positions]
    folds = [load_fold(FOLDS / f"fold_{i}", reference_columns) for i in range(1, 6)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = int(pathway_cfg["train"]["max_epochs"])
    dims = {name: folds[0]["dev"][name].shape[1] for name in ("mRNA", "CNV", "Proteomics")}

    results = {}
    jobs = [
        ("mean", mean_cfg, "pathway"),
        ("gene_vector", pathway_cfg, "pathway"),
        ("oscar", oscar_cfg, "oscar"),
        ("moxgate", moxgate_cfg, "moxgate"),
    ]
    for name, cfg, kind in jobs:
        rows = []
        for fold_id, fold in enumerate(folds, start=1):
            if kind == "pathway":
                row = run_pathway_on_fold(cfg, cohort, fold, device)
            elif kind == "moxgate":
                row = run_moxgate_fold(moxgate_cfg, dims, fold, device, max_epochs)
            else:
                row = run_oscar_fold(oscar_cfg, dims, fold, device, max_epochs)
            row["fold"] = fold_id
            rows.append(row)
            print(
                f"{name} fold {fold_id}: epoch {row['chosen_epoch']} "
                f"test macro-F1 {row['test']['f1_macro']:.4f}",
                flush=True,
            )
        results[name] = {"folds": rows, "summary": summarize(rows)}
        print(
            f"{name} mean macro-F1 {results[name]['summary']['f1_macro_mean']:.4f} "
            f"+/- {results[name]['summary']['f1_macro_std']:.4f}",
            flush=True,
        )

    report = {
        "protocol": {
            "outer_folds": "csv/processed/BRCA/5-fold",
            "held_out": "each fold validation file",
            "inner_val_fraction": 0.2,
            "inner_seed": 42,
            "max_epochs": max_epochs,
            "selection": "first maximum of inner validation macro-F1",
            "retrain": "all samples in the fold training file",
        },
        "models": results,
    }
    out_dir = Path("lightning_logs/brca_five_fold_matched")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved report to {report_path}", flush=True)


def run_pathway_on_fold(cfg, cohort, fold, device) -> dict:
    seed = int(cfg["train"]["seed"])
    mrna_pos = cohort["gene_positions"]
    cnv_pos = cohort["cnv_positions"]
    mrna_dev = fold["dev"]["mRNA"][:, mrna_pos]
    cnv_dev = fold["dev"]["CNV"][:, cnv_pos]
    protein_dev = fold["dev"]["Proteomics"]
    mrna_test = fold["test"]["mRNA"][:, mrna_pos]
    cnv_test = fold["test"]["CNV"][:, cnv_pos]
    protein_test = fold["test"]["Proteomics"]
    y_dev, y_test = fold["y_dev"], fold["y_test"]
    train_idx, val_idx = inner_split(y_dev, seed)
    batch_size = int(cfg["train"]["batch_size"])
    max_epochs = int(cfg["train"]["max_epochs"])

    def transform_all(fit_idx, parts):
        scaled = {}
        for key, dev_values, test_values in (
            ("mrna", mrna_dev, mrna_test),
            ("cnv", cnv_dev, cnv_test),
            ("protein", protein_dev, protein_test),
        ):
            scaler = StandardScaler().fit(dev_values[fit_idx])
            scaled[key] = {}
            for name, idx in parts.items():
                source = test_values if name == "test" else dev_values[idx]
                scaled[key][name] = scaler.transform(source).astype(np.float32)
        return scaled

    set_seed(seed)
    scaled = transform_all(train_idx, {"train": train_idx, "val": val_idx})
    train_loader = make_loader(
        scaled["mrna"]["train"], scaled["cnv"]["train"], scaled["protein"]["train"],
        y_dev[train_idx], batch_size, True, seed,
    )
    val_loader = make_loader(
        scaled["mrna"]["val"], scaled["cnv"]["val"], scaled["protein"]["val"],
        y_dev[val_idx], batch_size, False, seed,
    )
    model = build_model(cfg, cohort).to(device)
    optimizer = adamw(model, float(cfg["train"]["lr"]), float(cfg["train"]["weight_decay"]))
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
        scaled["mrna"]["train"], scaled["cnv"]["train"], scaled["protein"]["train"],
        y_dev, batch_size, True, seed,
    )
    test_loader = make_loader(
        scaled["mrna"]["test"], scaled["cnv"]["test"], scaled["protein"]["test"],
        y_test, batch_size, False, seed,
    )
    model = build_model(cfg, cohort).to(device)
    optimizer = adamw(model, float(cfg["train"]["lr"]), float(cfg["train"]["weight_decay"]))
    for _epoch in range(best_epoch):
        run_epoch(model, train_loader, optimizer, device)
    return {
        "n_inner_train": int(len(train_idx)),
        "n_inner_val": int(len(val_idx)),
        "n_retrain": int(len(y_dev)),
        "n_test": int(len(y_test)),
        "chosen_epoch": best_epoch,
        "inner_val_f1_macro": best_f1,
        "test": evaluate(model, test_loader, device),
    }


def main_moxgate_only() -> None:
    """Run MoXGATE on the same folds and merge it into the existing report."""
    report_path = Path("lightning_logs/brca_five_fold_matched/report.json")
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    moxgate_cfg = yaml.safe_load(open("configs/config_moxgate_brca.yaml"))
    pathway_cfg = yaml.safe_load(open("configs/config_pathway_tokens_brca.yaml"))
    configure_runtime()
    reference = _read_split_matrix(FOLDS / "fold_1", "BRCA_mRNA")
    reference_columns = {
        "mRNA": reference.index,
        "CNV": _read_split_matrix(FOLDS / "fold_1", "BRCA_CNV_1_channel").index,
        "Proteomics": _read_split_matrix(FOLDS / "fold_1", "BRCA_proteomics").index,
    }
    folds = [load_fold(FOLDS / f"fold_{i}", reference_columns) for i in range(1, 6)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = {name: folds[0]["dev"][name].shape[1] for name in ("mRNA", "CNV", "Proteomics")}
    max_epochs = int(pathway_cfg["train"]["max_epochs"])
    rows = []
    for fold_id, fold in enumerate(folds, start=1):
        row = run_moxgate_fold(moxgate_cfg, dims, fold, device, max_epochs)
        row["fold"] = fold_id
        rows.append(row)
        print(
            f"moxgate fold {fold_id}: epoch {row['chosen_epoch']} "
            f"test macro-F1 {row['test']['f1_macro']:.4f}",
            flush=True,
        )
    report["models"]["moxgate"] = {"folds": rows, "summary": summarize(rows)}
    summary = report["models"]["moxgate"]["summary"]
    print(
        f"moxgate mean macro-F1 {summary['f1_macro_mean']:.4f} "
        f"+/- {summary['f1_macro_std']:.4f}",
        flush=True,
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Updated {report_path}", flush=True)


def main_oscar_pathway() -> None:
    """Run OSCAR on Hallmark pathway means instead of full gene matrices."""
    report_path = Path("lightning_logs/brca_five_fold_matched/report.json")
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    oscar_cfg = yaml.safe_load(open("configs/config_self_attn_cross_attn_brca.yaml"))
    pathway_cfg = yaml.safe_load(open("configs/config_pathway_mean_brca.yaml"))
    configure_runtime()
    reference = _read_split_matrix(FOLDS / "fold_1", "BRCA_mRNA")
    reference_columns = {
        "mRNA": reference.index,
        "CNV": _read_split_matrix(FOLDS / "fold_1", "BRCA_CNV_1_channel").index,
        "Proteomics": _read_split_matrix(FOLDS / "fold_1", "BRCA_proteomics").index,
    }
    cohort = build_cohort(pathway_cfg)
    cohort["gene_positions"] = [int(reference_columns["mRNA"].get_loc(gene)) for gene in cohort["genes"]]
    cohort["cnv_positions"] = [int(reference_columns["CNV"].get_loc(gene)) for gene in cohort["genes"]]
    folds = [load_fold(FOLDS / f"fold_{i}", reference_columns) for i in range(1, 6)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = {name: folds[0]["dev"][name].shape[1] for name in ("mRNA", "CNV", "Proteomics")}
    max_epochs = int(pathway_cfg["train"]["max_epochs"])
    model_cfg = oscar_cfg["model"]
    train_cfg = oscar_cfg["train"]

    def build():
        return OscarPathwayModel(
            membership=cohort["membership"],
            mrna_index=torch.tensor(cohort["gene_positions"]),
            cnv_index=torch.tensor(cohort["cnv_positions"]),
            total_index=cohort["total_index"],
            total_mask=cohort["total_mask"],
            phospho_index=cohort["phospho_index"],
            phospho_mask=cohort["phospho_mask"],
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

    print(
        f"OSCAR pathway inputs: {len(cohort['pathway_names'])} CNV/mRNA pathways, "
        f"{cohort['total_index'].shape[0] * 4} protein program features, "
        f"{len(cohort['genes'])} genes",
        flush=True,
    )
    rows = []
    for fold_id, fold in enumerate(folds, start=1):
        row = run_oscar_fold(oscar_cfg, dims, fold, device, max_epochs, build=build)
        row["fold"] = fold_id
        rows.append(row)
        print(
            f"oscar_pathway fold {fold_id}: epoch {row['chosen_epoch']} "
            f"test macro-F1 {row['test']['f1_macro']:.4f}",
            flush=True,
        )
    report["models"]["oscar_pathway"] = {"folds": rows, "summary": summarize(rows)}
    summary = report["models"]["oscar_pathway"]["summary"]
    print(
        f"oscar_pathway mean macro-F1 {summary['f1_macro_mean']:.4f} "
        f"+/- {summary['f1_macro_std']:.4f}",
        flush=True,
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Updated {report_path}", flush=True)


def main_pathformer_pathways() -> None:
    """Rerun mean and gene-vector pathway models on Pathformer's 1497 pathways."""
    report_path = Path("lightning_logs/brca_five_fold_matched/report.json")
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    mean_cfg = yaml.safe_load(open("configs/config_pathway_mean_pathformer_brca.yaml"))
    vector_cfg = yaml.safe_load(open("configs/config_pathway_tokens_pathformer_brca.yaml"))
    configure_runtime()
    reference = _read_split_matrix(FOLDS / "fold_1", "BRCA_mRNA")
    reference_columns = {
        "mRNA": reference.index,
        "CNV": _read_split_matrix(FOLDS / "fold_1", "BRCA_CNV_1_channel").index,
        "Proteomics": _read_split_matrix(FOLDS / "fold_1", "BRCA_proteomics").index,
    }
    cohort = build_cohort(vector_cfg)
    cohort["gene_positions"] = [int(reference_columns["mRNA"].get_loc(gene)) for gene in cohort["genes"]]
    cohort["cnv_positions"] = [int(reference_columns["CNV"].get_loc(gene)) for gene in cohort["genes"]]
    folds = [load_fold(FOLDS / f"fold_{i}", reference_columns) for i in range(1, 6)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Pathformer pathways {len(cohort['pathway_names'])}, "
        f"genes {len(cohort['genes'])}, source {cohort['pathway_source']}",
        flush=True,
    )

    jobs = [
        ("mean_pathformer", mean_cfg),
        ("gene_vector_pathformer", vector_cfg),
    ]
    for name, cfg in jobs:
        rows = []
        for fold_id, fold in enumerate(folds, start=1):
            row = run_pathway_on_fold(cfg, cohort, fold, device)
            row["fold"] = fold_id
            rows.append(row)
            print(
                f"{name} fold {fold_id}: epoch {row['chosen_epoch']} "
                f"test macro-F1 {row['test']['f1_macro']:.4f}",
                flush=True,
            )
        report["models"][name] = {"folds": rows, "summary": summarize(rows)}
        summary = report["models"][name]["summary"]
        print(
            f"{name} mean macro-F1 {summary['f1_macro_mean']:.4f} "
            f"+/- {summary['f1_macro_std']:.4f}",
            flush=True,
        )

    report["pathformer_pathways"] = {
        "table": vector_cfg["data"]["pathformer_pathway_table"],
        "n_pathways": len(cohort["pathway_names"]),
        "n_genes": len(cohort["genes"]),
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Updated {report_path}", flush=True)


def main_moxgate_pathway() -> None:
    """Add the mean pathway token to MoXGATE and score the same five folds."""
    report_path = Path("lightning_logs/brca_five_fold_matched/report.json")
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    moxgate_cfg = yaml.safe_load(open("configs/config_moxgate_brca.yaml"))
    mean_cfg = yaml.safe_load(open("configs/config_pathway_mean_brca.yaml"))
    pathway_cfg = yaml.safe_load(open("configs/config_pathway_tokens_brca.yaml"))
    configure_runtime()
    reference = _read_split_matrix(FOLDS / "fold_1", "BRCA_mRNA")
    reference_columns = {
        "mRNA": reference.index,
        "CNV": _read_split_matrix(FOLDS / "fold_1", "BRCA_CNV_1_channel").index,
        "Proteomics": _read_split_matrix(FOLDS / "fold_1", "BRCA_proteomics").index,
    }
    cohort = build_cohort(pathway_cfg)
    cohort["gene_positions"] = [int(reference_columns["mRNA"].get_loc(gene)) for gene in cohort["genes"]]
    cohort["cnv_positions"] = [int(reference_columns["CNV"].get_loc(gene)) for gene in cohort["genes"]]
    folds = [load_fold(FOLDS / f"fold_{i}", reference_columns) for i in range(1, 6)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = {name: folds[0]["dev"][name].shape[1] for name in ("mRNA", "CNV", "Proteomics")}
    max_epochs = int(pathway_cfg["train"]["max_epochs"])
    model_cfg = moxgate_cfg["model"]
    train_cfg = moxgate_cfg["train"]

    def build():
        pathway = build_model(mean_cfg, cohort)
        return MoXGATEWithPathway(
            input_dims=dims,
            pathway=pathway,
            mrna_index=torch.tensor(cohort["gene_positions"]),
            cnv_index=torch.tensor(cohort["cnv_positions"]),
            num_classes=int(model_cfg["num_classes"]),
            latent_dim=int(model_cfg["latent_dim"]),
            classifier_dim=int(model_cfg.get("classifier_dim", model_cfg.get("fusion_hidden", 128))),
            attn_dropout=float(model_cfg.get("attn_dropout", model_cfg.get("dropout", 0.1))),
            classifier_dropout=float(model_cfg.get("classifier_dropout", 0.3)),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
            n_heads=int(model_cfg["n_heads"]),
            joint_n_heads=int(model_cfg["joint_n_heads"]),
            use_focal_loss=bool(model_cfg["use_focal_loss"]),
            focal_gamma=float(model_cfg["focal_gamma"]),
            lambda_modality=float(
                model_cfg.get("lambda_modality", model_cfg.get("modality_balance_weight", 1e-2))
            ),
            lambda_cross_frobenius=float(model_cfg.get("lambda_cross_frobenius", 1e-2)),
        )

    rows = []
    for fold_id, fold in enumerate(folds, start=1):
        row = run_moxgate_fold(moxgate_cfg, dims, fold, device, max_epochs, build=build)
        row["fold"] = fold_id
        rows.append(row)
        print(
            f"moxgate_pathway fold {fold_id}: epoch {row['chosen_epoch']} "
            f"test macro-F1 {row['test']['f1_macro']:.4f}",
            flush=True,
        )
    report["models"]["moxgate_pathway"] = {"folds": rows, "summary": summarize(rows)}
    summary = report["models"]["moxgate_pathway"]["summary"]
    print(
        f"moxgate_pathway mean macro-F1 {summary['f1_macro_mean']:.4f} "
        f"+/- {summary['f1_macro_std']:.4f}",
        flush=True,
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Updated {report_path}", flush=True)


if __name__ == "__main__":
    import sys
    if "--oscar-pathway" in sys.argv:
        main_oscar_pathway()
    elif "--pathformer-pathways" in sys.argv:
        main_pathformer_pathways()
    elif "--moxgate-pathway" in sys.argv:
        main_moxgate_pathway()
    elif "--moxgate-only" in sys.argv:
        main_moxgate_only()
    else:
        main()
