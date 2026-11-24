"""Matched 5-fold runner for the 6 paper baselines across BRCA, LUNG, and COADREAD:
  moxgate (paper-faithful), mot, deepmoic, mogcn, mogonet, mobrca.

Protocol matches train_token_oscar_matched.py:
  outer pre-split 5 folds -> inner 20% stratified val -> first-max val macro-F1
  -> retrain on full fold train -> score once on fold val.

Usage:
  uv run python train_baselines_matched.py --cohort BRCA --models moxgate,mot,deepmoic,mogcn,mogonet,mobrca
  uv run python train_baselines_matched.py --cohort all --models all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from baselines.models import (
    DeepMoICModel,
    MoBRCANetModel,
    MoGCNModel,
    MOGONETModel,
    MOTModel,
    MultiOmicAE,
    build_snf_adj,
    cosine_knn_adj,
)
from pathway_data import _read_split_matrix
from train_five_fold import inner_split, run_moxgate_fold, summarize
from train_oscar_matched import evaluate as evaluate_batch, make_loader
from train_pathway_tokens import configure_runtime, set_seed
from train_token_oscar_matched import COHORTS, MATCHED_MAX_EPOCHS, _stems, load_fold

ALL_MODELS = ("moxgate", "mot", "deepmoic", "mogcn", "mogonet", "mobrca")
OUT_DIR = Path("lightning_logs/baselines_matched")


def _metrics(logits: torch.Tensor, y: np.ndarray) -> dict:
    preds = logits.argmax(dim=-1).detach().cpu().numpy()
    label_ids = list(range(logits.shape[-1]))
    return {
        "f1_macro": float(f1_score(y, preds, average="macro", labels=label_ids)),
        "accuracy": float(accuracy_score(y, preds)),
        "f1_per_class": f1_score(y, preds, average=None, labels=label_ids).tolist(),
    }


def _scale_fold(fold: dict, fit_idx: np.ndarray, parts: dict) -> dict:
    scaled = {}
    for name in ("mRNA", "CNV", "Proteomics"):
        scaler = StandardScaler().fit(fold["dev"][name][fit_idx])
        scaled[name] = {}
        for part, idx in parts.items():
            src = fold["test"][name] if part == "test" else fold["dev"][name][idx]
            scaled[name][part] = scaler.transform(src).astype(np.float32)
    return scaled


def _topk_var(x: np.ndarray, k: int) -> np.ndarray:
    if x.shape[1] <= k:
        return np.arange(x.shape[1])
    var = x.var(axis=0)
    return np.sort(np.argpartition(-var, kth=k - 1)[:k])


def _run_batch_model_fold(
    fold: dict,
    device: torch.device,
    build_fn,
    max_epochs: int = MATCHED_MAX_EPOCHS,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    seed: int = 42,
    topk: dict[str, int] | None = None,
) -> dict:
    y_dev, y_test = fold["y_dev"], fold["y_test"]
    train_idx, val_idx = inner_split(y_dev, seed)

    def prep(fit_idx, parts):
        sc = _scale_fold(fold, fit_idx, parts)
        if topk:
            for name, k in topk.items():
                cols = _topk_var(sc[name]["train"], k)
                for p in sc[name]:
                    sc[name][p] = sc[name][p][:, cols]
        return sc

    def run_epoch(model, loader, opt):
        model.train()
        for mrna, cnv, prot, y in loader:
            if mrna.shape[0] < 2:  # BatchNorm (moBRCA-net) needs >1 sample in train mode
                continue
            opt.zero_grad(set_to_none=True)
            logits = model({"mRNA": mrna.to(device), "CNV": cnv.to(device), "Proteomics": prot.to(device)})
            loss = F.cross_entropy(logits, y.to(device))
            loss.backward()
            opt.step()

    set_seed(seed)
    sc = prep(train_idx, {"train": train_idx, "val": val_idx})
    dims = {k: sc[k]["train"].shape[1] for k in ("mRNA", "CNV", "Proteomics")}
    tr_loader = make_loader(sc["mRNA"]["train"], sc["CNV"]["train"], sc["Proteomics"]["train"], y_dev[train_idx], batch_size, True, seed)
    va_loader = make_loader(sc["mRNA"]["val"], sc["CNV"]["val"], sc["Proteomics"]["val"], y_dev[val_idx], batch_size, False, seed)
    model = build_fn(dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_epoch, best_f1 = 1, -1.0
    for ep in range(1, max_epochs + 1):
        run_epoch(model, tr_loader, opt)
        score = evaluate_batch(model, va_loader, device)["f1_macro"]
        if score > best_f1:
            best_f1, best_epoch = score, ep

    set_seed(seed)
    dev_idx = np.arange(len(y_dev))
    sc = prep(dev_idx, {"train": dev_idx, "test": None})
    dims = {k: sc[k]["train"].shape[1] for k in ("mRNA", "CNV", "Proteomics")}
    tr_loader = make_loader(sc["mRNA"]["train"], sc["CNV"]["train"], sc["Proteomics"]["train"], y_dev, batch_size, True, seed)
    te_loader = make_loader(sc["mRNA"]["test"], sc["CNV"]["test"], sc["Proteomics"]["test"], y_test, batch_size, False, seed)
    model = build_fn(dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(best_epoch):
        run_epoch(model, tr_loader, opt)
    return {
        "chosen_epoch": best_epoch,
        "inner_val_f1_macro": best_f1,
        "test": evaluate_batch(model, te_loader, device),
    }


def _run_snf_gcn_fold(
    fold: dict,
    num_classes: int,
    device: torch.device,
    deep: bool,
    ae_epochs: int = 40,
    gcn_epochs: int = MATCHED_MAX_EPOCHS,
    seed: int = 42,
) -> dict:
    """Shared AE + SNF + (MoGCN or DeepMoIC) transductive matched-5-fold runner."""
    y_dev, y_test = fold["y_dev"], fold["y_test"]
    train_idx, val_idx = inner_split(y_dev, seed)

    def fit_ae_and_snf(sc, split_a: str, split_b: str):
        views_tr = [sc[k][split_a] for k in ("mRNA", "CNV", "Proteomics")]
        views_all = [np.concatenate([sc[k][split_a], sc[k][split_b]], axis=0) for k in ("mRNA", "CNV", "Proteomics")]
        dims = {k: sc[k][split_a].shape[1] for k in ("mRNA", "CNV", "Proteomics")}
        ae = MultiOmicAE(dims, latent_dim=128).to(device)
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-4)
        tr_t = {k: torch.from_numpy(sc[k][split_a]).to(device) for k in dims}
        ae.train()
        for _ in range(ae_epochs):
            opt.zero_grad(set_to_none=True)
            _, recons = ae(tr_t)
            loss = sum(F.mse_loss(recons[k], tr_t[k]) for k in dims)
            loss.backward()
            opt.step()
        ae.eval()
        with torch.no_grad():
            all_t = {k: torch.from_numpy(v).to(device) for k, v in zip(("mRNA", "CNV", "Proteomics"), views_all)}
            z_all = ae.encode(all_t)
        adj = build_snf_adj(views_all, k=20, t=10).to(device)
        return z_all, adj

    def make_gcn():
        return DeepMoICModel(128, num_classes).to(device) if deep else MoGCNModel(128, num_classes).to(device)

    set_seed(seed)
    sc = _scale_fold(fold, train_idx, {"train": train_idx, "val": val_idx})
    z_all, adj = fit_ae_and_snf(sc, "train", "val")
    n_tr = len(train_idx)
    y_tr_t = torch.from_numpy(y_dev[train_idx]).long().to(device)
    gcn = make_gcn()
    opt = torch.optim.Adam(gcn.parameters(), lr=1e-2, weight_decay=5e-4)
    best_epoch, best_f1 = 1, -1.0
    for ep in range(1, gcn_epochs + 1):
        gcn.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(gcn(z_all, adj)[:n_tr], y_tr_t)
        loss.backward()
        opt.step()
        gcn.eval()
        with torch.no_grad():
            score = _metrics(gcn(z_all, adj)[n_tr:], y_dev[val_idx])["f1_macro"]
        if score > best_f1:
            best_f1, best_epoch = score, ep

    set_seed(seed)
    dev_idx = np.arange(len(y_dev))
    sc = _scale_fold(fold, dev_idx, {"train": dev_idx, "test": None})
    z_all, adj = fit_ae_and_snf(sc, "train", "test")
    n_dev = len(y_dev)
    y_dev_t = torch.from_numpy(y_dev).long().to(device)
    gcn = make_gcn()
    opt = torch.optim.Adam(gcn.parameters(), lr=1e-2, weight_decay=5e-4)
    for _ in range(best_epoch):
        gcn.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(gcn(z_all, adj)[:n_dev], y_dev_t)
        loss.backward()
        opt.step()
    gcn.eval()
    with torch.no_grad():
        test_m = _metrics(gcn(z_all, adj)[n_dev:], y_test)
    return {"chosen_epoch": best_epoch, "inner_val_f1_macro": best_f1, "test": test_m}


def _run_mogonet_fold(fold: dict, num_classes: int, device: torch.device, max_epochs: int = MATCHED_MAX_EPOCHS, seed: int = 42) -> dict:
    y_dev, y_test = fold["y_dev"], fold["y_test"]
    train_idx, val_idx = inner_split(y_dev, seed)

    def make_graphs(sc, a: str, b: str):
        xs, adjs = {}, {}
        for k in ("mRNA", "CNV", "Proteomics"):
            arr = np.concatenate([sc[k][a], sc[k][b]], axis=0)
            xs[k] = torch.from_numpy(arr).to(device)
            adjs[k] = cosine_knn_adj(arr, k=10).to(device)
        return xs, adjs

    set_seed(seed)
    sc = _scale_fold(fold, train_idx, {"train": train_idx, "val": val_idx})
    dims = {k: sc[k]["train"].shape[1] for k in ("mRNA", "CNV", "Proteomics")}
    xs, adjs = make_graphs(sc, "train", "val")
    n_tr = len(train_idx)
    y_tr_t = torch.from_numpy(y_dev[train_idx]).long().to(device)
    model = MOGONETModel(dims, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_epoch, best_f1 = 1, -1.0
    for ep in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        fused, per_omic = model(xs, adjs)
        loss = F.cross_entropy(fused[:n_tr], y_tr_t) + sum(F.cross_entropy(v[:n_tr], y_tr_t) for v in per_omic.values())
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            score = _metrics(model(xs, adjs)[0][n_tr:], y_dev[val_idx])["f1_macro"]
        if score > best_f1:
            best_f1, best_epoch = score, ep

    set_seed(seed)
    dev_idx = np.arange(len(y_dev))
    sc = _scale_fold(fold, dev_idx, {"train": dev_idx, "test": None})
    xs, adjs = make_graphs(sc, "train", "test")
    n_dev = len(y_dev)
    y_dev_t = torch.from_numpy(y_dev).long().to(device)
    model = MOGONETModel(dims, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(best_epoch):
        model.train()
        opt.zero_grad(set_to_none=True)
        fused, per_omic = model(xs, adjs)
        loss = F.cross_entropy(fused[:n_dev], y_dev_t) + sum(F.cross_entropy(v[:n_dev], y_dev_t) for v in per_omic.values())
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        test_m = _metrics(model(xs, adjs)[0][n_dev:], y_test)
    return {"chosen_epoch": best_epoch, "inner_val_f1_macro": best_f1, "test": test_m}


MOXGATE_CONFIGS = {
    "BRCA": "configs/config_moxgate_brca.yaml",
    "LUNG": "configs/config_moxgate_lung.yaml",
    "COADREAD": "configs/config_moxgate_crc.yaml",
}


def run_baseline_on_cohort(cohort_name: str, model_name: str, folds: list[dict], dims: dict[str, int], num_classes: int, device: torch.device) -> dict:
    rows = []
    for fold_id, fold in enumerate(folds, start=1):
        if model_name == "moxgate":
            cfg = yaml.safe_load(open(MOXGATE_CONFIGS[cohort_name]))
            assert int(cfg["model"]["num_classes"]) == num_classes, f"{cohort_name} moxgate num_classes mismatch"
            row = run_moxgate_fold(cfg, dims, fold, device, MATCHED_MAX_EPOCHS)
        elif model_name == "mot":
            row = _run_batch_model_fold(fold, device, lambda d: MOTModel(d, num_classes), lr=2e-4, weight_decay=1e-4)
        elif model_name == "mobrca":
            row = _run_batch_model_fold(
                fold,
                device,
                lambda d: MoBRCANetModel(d, num_classes),
                lr=1e-3,
                weight_decay=1e-4,
                topk={"mRNA": 1000, "CNV": 1000},
            )
        elif model_name == "mogcn":
            row = _run_snf_gcn_fold(fold, num_classes, device, deep=False)
        elif model_name == "deepmoic":
            row = _run_snf_gcn_fold(fold, num_classes, device, deep=True)
        elif model_name == "mogonet":
            row = _run_mogonet_fold(fold, num_classes, device)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        row["fold"] = fold_id
        rows.append(row)
        print(
            f"[{cohort_name}][{model_name}] fold {fold_id}: ep {row['chosen_epoch']} "
            f"F1 {row['test']['f1_macro']:.4f} acc {row['test']['accuracy']:.4f}",
            flush=True,
        )
    return {"folds": rows, "summary": summarize(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="BRCA", help="all or comma-separated BRCA,LUNG,COADREAD")
    parser.add_argument("--models", default="all", help="all or comma-separated moxgate,mot,deepmoic,mogcn,mogonet,mobrca")
    args = parser.parse_args()

    configure_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohorts = list(COHORTS.keys()) if args.cohort == "all" else [c.strip().upper() for c in args.cohort.split(",") if c.strip()]
    models = list(ALL_MODELS) if args.models == "all" else [m.strip().lower() for m in args.models.split(",") if m.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cohort_name in cohorts:
        spec = COHORTS[cohort_name]
        oscar_cfg = yaml.safe_load(open(spec["oscar"]))
        num_classes = int(oscar_cfg["model"]["num_classes"])
        prefix = spec["prefix"]
        stems = _stems(prefix)
        folds_dir = spec["folds"]
        ref_cols = {
            "mRNA": _read_split_matrix(folds_dir / "fold_1", stems["mRNA"]).index,
            "CNV": _read_split_matrix(folds_dir / "fold_1", stems["CNV"]).index,
            "Proteomics": _read_split_matrix(folds_dir / "fold_1", stems["Proteomics"]).index,
        }
        folds = [load_fold(folds_dir / f"fold_{i}", ref_cols, prefix) for i in range(1, 6)]
        dims = {k: folds[0]["dev"][k].shape[1] for k in ("mRNA", "CNV", "Proteomics")}

        out_path = OUT_DIR / f"{cohort_name.lower()}_report.json"
        data = json.load(open(out_path)) if out_path.exists() else {"models": {}}
        for m in models:
            data["models"][m] = run_baseline_on_cohort(cohort_name, m, folds, dims, num_classes, device)
            s = data["models"][m]["summary"]
            print(
                f"[{cohort_name}][{m}] mean F1 {100*s['f1_macro_mean']:.2f}+/-{100*s['f1_macro_std']:.2f} "
                f"acc {100*s['accuracy_mean']:.2f}+/-{100*s['accuracy_std']:.2f}",
                flush=True,
            )
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
