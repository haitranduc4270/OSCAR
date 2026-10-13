"""First probe for shared vs modality-specific latents on BRCA.

Each omic is encoded by an MLP and split into a concordant head and a
private head. The full variant adds InfoNCE on the adjacent Central Dogma
edges (CNV-mRNA, mRNA-RPPA) plus an orthogonality penalty. The shuffle
variant trains the same losses after breaking patient alignment.

The readout is cross-modal: can the CNV concordant latent predict ERBB2
mRNA better than the CNV private latent, after removing subtype means?
MKI67 expression predicted from CNV is the negative biological control.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
FOLD_ROOT = ROOT / "csv" / "processed" / "BRCA" / "5-fold"
OUT_DIR = ROOT / "lightning_logs" / "disentangle_probe"

ORDER = ("CNV", "mRNA", "Proteomics")
STEMS = {
    "CNV": "BRCA_CNV_1_channel",
    "mRNA": "BRCA_mRNA",
    "Proteomics": "BRCA_proteomics",
}
LABELS = {0: "LumA", 1: "LumB", 2: "Basal", 3: "HER2"}

# Cross-modal questions. Source is the latent's modality, target is another
# measurement of the same patient. Within-modality rows are included only
# where the split itself is the question (mRNA proliferation).
DECODES = [
    ("erbb2_mrna_from_cnv", "mRNA", "ERBB2", "CNV"),
    ("her2_from_mrna", "Proteomics", "HER2", "mRNA"),
    ("her2_from_cnv", "Proteomics", "HER2", "CNV"),
    ("her2p_from_mrna", "Proteomics", "HER2_pY1248", "mRNA"),
    ("eralpha_from_mrna", "Proteomics", "ERALPHA", "mRNA"),
    ("mki67_from_cnv", "mRNA", "MKI67", "CNV"),
    ("aurka_from_cnv", "mRNA", "AURKA", "CNV"),
    ("mki67_from_mrna", "mRNA", "MKI67", "mRNA"),
    ("esr1_from_prot", "mRNA", "ESR1", "Proteomics"),
]

CEILINGS = [
    ("erbb2_cnv_mrna", "CNV", "ERBB2", "mRNA", "ERBB2"),
    ("erbb2_mrna_her2", "mRNA", "ERBB2", "Proteomics", "HER2"),
    ("erbb2_mrna_her2p", "mRNA", "ERBB2", "Proteomics", "HER2_pY1248"),
    ("mki67_cnv_mrna", "CNV", "MKI67", "mRNA", "MKI67"),
    ("esr1_mrna_eralpha", "mRNA", "ESR1", "Proteomics", "ERALPHA"),
]

SALIENCY = [
    ("erbb2_mrna_from_cnv", "CNV"),
    ("her2_from_mrna", "mRNA"),
    ("mki67_from_cnv", "CNV"),
]

LAM_NCE = 0.2
LAM_ORTHO = 1.0
LAM_CROSS = 1.0
TEMPERATURE = 0.2


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SplitEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent: int, dropout: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_c = nn.Linear(hidden, latent)
        self.head_p = nn.Linear(hidden, latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.head_c(h), self.head_p(h)


class DisentangledOmics(nn.Module):
    def __init__(self, dims: dict[str, int], n_classes: int, hidden: int, latent: int, dropout: float):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {name: SplitEncoder(dims[name], hidden, latent, dropout) for name in ORDER}
        )
        fused = latent * 2 * len(ORDER)
        self.classifier = nn.Sequential(
            nn.Linear(fused, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {name: self.encoders[name](batch[name]) for name in ORDER}

    def forward(self, batch: dict[str, torch.Tensor]):
        encoded = self.encode(batch)
        parts = []
        for name in ORDER:
            zc, zp = encoded[name]
            parts.extend((zc, zp))
        logits = self.classifier(torch.cat(parts, dim=-1))
        return logits, encoded


class OmicsDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], y: np.ndarray):
        self.arrays = {name: torch.from_numpy(np.ascontiguousarray(values)) for name, values in arrays.items()}
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {name: values[idx] for name, values in self.arrays.items()}
        item["label"] = self.y[idx]
        return item


def read_omic(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    if frame.isna().any().any():
        raise ValueError(f"Missing values in {path}")
    return frame


def load_fold(fold: int) -> dict:
    folder = FOLD_ROOT / f"fold_{fold}"
    frames = {}
    for name, stem in STEMS.items():
        frames[name] = {
            "train": read_omic(folder / f"{stem}_train.csv"),
            "val": read_omic(folder / f"{stem}_val.csv"),
        }
    y_train = pd.read_csv(folder / "BRCA_label_num_train.csv").iloc[:, 0].to_numpy()
    y_val = pd.read_csv(folder / "BRCA_label_num_val.csv").iloc[:, 0].to_numpy()

    samples = {
        "train": list(frames["mRNA"]["train"].columns),
        "val": list(frames["mRNA"]["val"].columns),
    }
    for name in ORDER:
        for split in ("train", "val"):
            missing = [sample for sample in samples[split] if sample not in frames[name][split].columns]
            if missing:
                raise KeyError(f"{name} {split} missing {len(missing)} samples")
            frames[name][split] = frames[name][split].loc[:, samples[split]]

    gene_names = {name: frames[name]["train"].index.to_numpy() for name in ORDER}
    raw = {}
    scaled = {}
    for split in ("train", "val"):
        raw[split] = {}
        scaled[split] = {}
    for name in ORDER:
        train = frames[name]["train"].T.to_numpy(dtype=np.float64)
        val = frames[name]["val"].T.to_numpy(dtype=np.float64)
        scaler = StandardScaler()
        raw["train"][name] = train.astype(np.float32)
        raw["val"][name] = val.astype(np.float32)
        scaled["train"][name] = scaler.fit_transform(train).astype(np.float32)
        scaled["val"][name] = scaler.transform(val).astype(np.float32)

    return {
        "y_train": y_train.astype(np.int64),
        "y_val": y_val.astype(np.int64),
        "raw": raw,
        "scaled": scaled,
        "genes": gene_names,
    }


def gene_index(genes: np.ndarray, symbol: str) -> int:
    hits = np.where(genes == symbol)[0]
    if len(hits) != 1:
        raise KeyError(f"{symbol} not found uniquely ({len(hits)} hits)")
    return int(hits[0])


def column(data: dict, split: str, modality: str, symbol: str) -> np.ndarray:
    idx = gene_index(data["genes"][modality], symbol)
    return data["raw"][split][modality][:, idx].astype(np.float64)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ceilings(data: dict) -> dict[str, float]:
    out = {}
    for name, src_mod, src_gene, tgt_mod, tgt_gene in CEILINGS:
        x_tr = column(data, "train", src_mod, src_gene).reshape(-1, 1)
        y_tr = column(data, "train", tgt_mod, tgt_gene)
        x_va = column(data, "val", src_mod, src_gene).reshape(-1, 1)
        y_va = column(data, "val", tgt_mod, tgt_gene)
        reg = LinearRegression().fit(x_tr, y_tr)
        pred = reg.predict(x_va)
        out[name] = {
            "val_r": pearson(pred, y_va),
            "val_r2": float(reg.score(x_va, y_va)),
            "train_r": pearson(x_tr[:, 0], y_tr),
        }
    return out


def subtype_residual(y_train: np.ndarray, y_val: np.ndarray, lab_train: np.ndarray, lab_val: np.ndarray):
    means = {int(lab): float(y_train[lab_train == lab].mean()) for lab in np.unique(lab_train)}
    global_mean = float(y_train.mean())
    r_train = y_train - np.array([means[int(lab)] for lab in lab_train])
    r_val = y_val - np.array([means.get(int(lab), global_mean) for lab in lab_val])
    return r_train, r_val


def info_nce(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.T / TEMPERATURE
    labels = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def matched_cos_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1).pow(2).mean()


def alignment_loss(encoded: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    zc = {name: encoded[name][0] for name in ORDER}
    zp = {name: encoded[name][1] for name in ORDER}
    nce = 0.5 * (info_nce(zc["CNV"], zc["mRNA"]) + info_nce(zc["mRNA"], zc["Proteomics"]))
    ortho = torch.stack([matched_cos_sq(zc[name], zp[name]) for name in ORDER]).mean()
    cross_pairs = (
        (zp["CNV"], zc["mRNA"]),
        (zp["mRNA"], zc["CNV"]),
        (zp["mRNA"], zc["Proteomics"]),
        (zp["Proteomics"], zc["mRNA"]),
        (zp["CNV"], zc["Proteomics"]),
        (zp["Proteomics"], zc["CNV"]),
    )
    cross = torch.stack([matched_cos_sq(a, b) for a, b in cross_pairs]).mean()
    return {"nce": nce, "ortho": ortho, "cross": cross}


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def predict_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch_to_device(batch, device)
        logits, _ = model(batch)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels)


@torch.no_grad()
def collect_latents(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, dict[str, np.ndarray]]:
    model.eval()
    bags = {name: {"c": [], "p": []} for name in ORDER}
    for batch in loader:
        batch = batch_to_device(batch, device)
        _, encoded = model(batch)
        for name in ORDER:
            zc, zp = encoded[name]
            bags[name]["c"].append(zc.cpu().numpy())
            bags[name]["p"].append(zp.cpu().numpy())
    return {
        name: {"c": np.concatenate(parts["c"]), "p": np.concatenate(parts["p"])}
        for name, parts in bags.items()
    }


@torch.no_grad()
def matched_cosine(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a_t = F.normalize(torch.from_numpy(a), dim=-1)
    b_t = F.normalize(torch.from_numpy(b), dim=-1)
    matched = float((a_t * b_t).sum(dim=-1).mean())
    shifted = float((a_t * b_t.roll(1, dims=0)).sum(dim=-1).mean())
    return {"matched": matched, "shifted": shifted}


def decode_r(z_train: np.ndarray, y_train: np.ndarray, z_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    scaler = StandardScaler()
    z_tr = scaler.fit_transform(z_train)
    z_va = scaler.transform(z_val)
    reg = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(z_tr, y_train)
    pred = reg.predict(z_va)
    scale = np.maximum(scaler.scale_, 1e-8)
    return {
        "r": pearson(pred, y_val),
        "r2": float(reg.score(z_va, y_val)),
        "coef": (reg.coef_ / scale).astype(np.float64),
    }


def run_decodes(data: dict, z_train: dict, z_val: dict) -> dict:
    out = {}
    y_lab_tr, y_lab_va = data["y_train"], data["y_val"]
    for name, tgt_mod, tgt_gene, src_mod in DECODES:
        y_tr = column(data, "train", tgt_mod, tgt_gene)
        y_va = column(data, "val", tgt_mod, tgt_gene)
        r_tr, r_va = subtype_residual(y_tr, y_va, y_lab_tr, y_lab_va)
        block = {}
        for head in ("c", "p"):
            raw = decode_r(z_train[src_mod][head], y_tr, z_val[src_mod][head], y_va)
            resid = decode_r(z_train[src_mod][head], r_tr, z_val[src_mod][head], r_va)
            block[head] = {
                "raw_r": raw["r"],
                "raw_r2": raw["r2"],
                "resid_r": resid["r"],
                "resid_r2": resid["r2"],
            }
            block[head + "_coef"] = resid["coef"]
        block["resid_gap_c_minus_p"] = block["c"]["resid_r"] - block["p"]["resid_r"]
        out[name] = block
    return out


def top_genes(grad: np.ndarray, genes: np.ndarray, k: int = 8) -> list[dict]:
    order = np.argsort(-grad)
    rows = []
    for rank, idx in enumerate(order[:k], start=1):
        rows.append({"rank": rank, "gene": str(genes[idx]), "grad": float(grad[idx])})
    return rows


def direction_saliency(
    model: DisentangledOmics,
    modality: str,
    head: str,
    values: np.ndarray,
    coef: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(values)).to(device)
    x.requires_grad_(True)
    zc, zp = model.encoders[modality](x)
    z = zc if head == "c" else zp
    direction = torch.as_tensor(coef, device=device, dtype=z.dtype)
    pred = (z * direction).sum()
    grad = torch.autograd.grad(pred, x)[0]
    return grad.detach().abs().mean(dim=0).cpu().numpy()


def gene_rank(grad: np.ndarray, genes: np.ndarray, symbol: str) -> dict:
    idx = gene_index(genes, symbol)
    order = np.argsort(-grad)
    rank = int(np.where(order == idx)[0][0]) + 1
    return {"gene": symbol, "rank": rank, "n": int(len(genes)), "percentile": 100.0 * (1.0 - (rank - 1) / len(genes))}


def shuffle_modalities(arrays: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {name: values.copy() for name, values in arrays.items()}
    for name in ("CNV", "Proteomics"):
        out[name] = out[name][rng.permutation(len(out[name]))]
    return out


def make_loader(arrays: dict[str, np.ndarray], y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        OmicsDataset(arrays, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def train_variant(
    data: dict,
    variant: str,
    fold: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden: int,
    latent: int,
    dropout: float,
) -> dict:
    align = variant in {"full", "shuffle"}
    seed = 42 + fold
    set_seed(seed)
    train_x = data["scaled"]["train"]
    val_x = data["scaled"]["val"]
    if variant == "shuffle":
        train_x = shuffle_modalities(train_x, seed + 10007)
        val_x = shuffle_modalities(val_x, seed + 20011)

    train_loader = make_loader(train_x, data["y_train"], batch_size, True, seed)
    val_loader = make_loader(val_x, data["y_val"], batch_size, False, seed)
    # Probes always read the true patient alignment. The shuffle variant is
    # scored on mismatched inputs so a leaked shortcut cannot look like concordance.
    probe_train_x = train_x
    probe_val_x = val_x

    dims = {name: data["scaled"]["train"][name].shape[1] for name in ORDER}
    model = DisentangledOmics(dims, n_classes=4, hidden=hidden, latent=latent, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_f1 = -1.0
    best_acc = None
    best_state = None
    best_epoch = 0
    stale = 0
    last_parts = {}

    for epoch in range(1, max_epochs + 1):
        model.train()
        totals = {"ce": 0.0, "nce": 0.0, "ortho": 0.0, "cross": 0.0, "n": 0}
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            logits, encoded = model(batch)
            ce = F.cross_entropy(logits, batch["label"])
            parts = alignment_loss(encoded) if align else None
            loss = ce
            if parts is not None:
                loss = ce + LAM_NCE * parts["nce"] + LAM_ORTHO * parts["ortho"] + LAM_CROSS * parts["cross"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bs = int(batch["label"].shape[0])
            totals["ce"] += float(ce.detach()) * bs
            totals["n"] += bs
            if parts is not None:
                for key in ("nce", "ortho", "cross"):
                    totals[key] += float(parts[key].detach()) * bs

        preds, labels = predict_labels(model, val_loader, device)
        val_f1 = float(f1_score(labels, preds, average="macro"))
        val_acc = float(accuracy_score(labels, preds))
        denom = max(totals["n"], 1)
        last_parts = {key: totals[key] / denom for key in ("ce", "nce", "ortho", "cross")}
        print(
            f"fold={fold} variant={variant} ep={epoch:02d} "
            f"ce={last_parts['ce']:.3f} nce={last_parts['nce']:.3f} "
            f"ortho={last_parts['ortho']:.3f} cross={last_parts['cross']:.3f} "
            f"val_f1={val_f1:.3f} val_acc={val_acc:.3f}",
            flush=True,
        )
        if val_f1 > best_f1 + 1e-3:
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    probe_train_loader = make_loader(probe_train_x, data["y_train"], batch_size, False, seed)
    probe_val_loader = make_loader(probe_val_x, data["y_val"], batch_size, False, seed)
    z_train = collect_latents(model, probe_train_loader, device)
    z_val = collect_latents(model, probe_val_loader, device)
    decodes = run_decodes(data, z_train, z_val)

    saliency = {}
    for probe_name, modality in SALIENCY:
        saliency[probe_name] = {}
        genes = data["genes"][modality]
        values = probe_val_x[modality]
        for head in ("c", "p"):
            coef = decodes[probe_name].pop(head + "_coef")
            grad = direction_saliency(model, modality, head, values, coef, device)
            entry = {"top": top_genes(grad, genes)}
            if probe_name == "erbb2_mrna_from_cnv":
                entry["ERBB2"] = gene_rank(grad, genes, "ERBB2")
            if probe_name == "her2_from_mrna":
                entry["ERBB2"] = gene_rank(grad, genes, "ERBB2")
            if probe_name == "mki67_from_cnv":
                entry["MKI67"] = gene_rank(grad, genes, "MKI67")
            saliency[probe_name][head] = entry
    for probe_name, _, _, _ in DECODES:
        for head in ("c", "p"):
            decodes[probe_name].pop(head + "_coef", None)

    alignment = {
        "cnv_mrna_c": matched_cosine(z_val["CNV"]["c"], z_val["mRNA"]["c"]),
        "mrna_prot_c": matched_cosine(z_val["mRNA"]["c"], z_val["Proteomics"]["c"]),
        "cnv_mrna_p": matched_cosine(z_val["CNV"]["p"], z_val["mRNA"]["p"]),
    }
    result = {
        "fold": fold,
        "variant": variant,
        "best_epoch": best_epoch,
        "val_f1": best_f1,
        "val_acc": best_acc,
        "train_losses_last": last_parts,
        "alignment": alignment,
        "decodes": decodes,
        "saliency": saliency,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def summarize(rows: list[dict]) -> dict:
    frame_rows = []
    for row in rows:
        base = {
            "fold": row["fold"],
            "variant": row["variant"],
            "val_f1": row["val_f1"],
            "val_acc": row["val_acc"],
            "align_cnv_mrna": row["alignment"]["cnv_mrna_c"]["matched"],
            "align_cnv_mrna_shift": row["alignment"]["cnv_mrna_c"]["shifted"],
            "align_mrna_prot": row["alignment"]["mrna_prot_c"]["matched"],
        }
        for name, block in row["decodes"].items():
            base[f"{name}_c"] = block["c"]["resid_r"]
            base[f"{name}_p"] = block["p"]["resid_r"]
            base[f"{name}_gap"] = block["resid_gap_c_minus_p"]
        erbb2 = row["saliency"]["erbb2_mrna_from_cnv"]
        base["erbb2_cnv_rank_c"] = erbb2["c"]["ERBB2"]["rank"]
        base["erbb2_cnv_rank_p"] = erbb2["p"]["ERBB2"]["rank"]
        frame_rows.append(base)
    frame = pd.DataFrame(frame_rows)
    numeric = [col for col in frame.columns if col not in {"fold", "variant"}]
    summary = (
        frame.groupby("variant")[numeric]
        .agg(["mean", "std"])
        .sort_index()
    )
    return {"table": frame, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--variants", nargs="+", default=["ce", "full", "shuffle"])
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.4)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    rows = []
    ceilings_by_fold = {}
    for fold in args.folds:
        print(f"loading fold {fold}", flush=True)
        data = load_fold(fold)
        ceilings_by_fold[str(fold)] = ceilings(data)
        print("ceilings", json.dumps(ceilings_by_fold[str(fold)]), flush=True)
        for variant in args.variants:
            row = train_variant(
                data,
                variant,
                fold,
                device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                hidden=args.hidden,
                latent=args.latent,
                dropout=args.dropout,
            )
            rows.append(row)
            path = OUT_DIR / "results.json"
            payload = {"ceilings": ceilings_by_fold, "runs": rows, "labels": LABELS}
            path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
            print(
                f"saved fold={fold} variant={variant} "
                f"f1={row['val_f1']:.3f} acc={row['val_acc']:.3f}",
                flush=True,
            )
        del data

    packed = summarize(rows)
    packed["table"].to_csv(OUT_DIR / "per_fold.csv", index=False)
    summary = packed["summary"]
    summary.to_csv(OUT_DIR / "summary.csv")
    print(summary.round(3).to_string(), flush=True)
    print(f"wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
