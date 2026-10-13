"""Compare omic-vector OSCAR with gene-token sequences on BRCA.

Four models see the same gene panel. A gene is included only when CNV, mRNA,
and a mapped RPPA antibody all exist, so the within-gene triple is observed.
Genes stay in CNV row order, which follows genomic coordinates from
chromosome 1 through Y.

Variants
    A  omic vectors. OSCAR encoders, ordered cross-attention, self-attention
       on three omic tokens.
    B  one token per gene. The three values are projected together, so there
       is no within-gene sequence. Genes sit in genomic order.
    C  within each gene, tokens run CNV then mRNA then protein. Genes sit in
       genomic order.
    D  same stack as C, but the gene sequence is a fixed shuffle. Position
       embeddings stay on the slots, gene embeddings move with the genes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from models.gene_sequence import GeneSequenceModel
from models.self_attn_fusion import SelfAttentionFusionModel
from train import ValAccAtBestF1Callback

ROOT = Path(__file__).resolve().parent
FOLD_ROOT = ROOT / "csv" / "processed" / "BRCA" / "5-fold"
GENE_INFO = ROOT / "csv" / "gene_coords" / "Homo_sapiens.gene_info.gz"
OUT_DIR = ROOT / "lightning_logs" / "gene_token_order"

# None marks an antibody that binds more than one gene. Those measurements
# are dropped so a single value is never copied onto several loci.
ANTIBODY_TO_GENE = {
    "1433EPSILON": "YWHAE",
    "4EBP1": "EIF4EBP1",
    "4EBP1_pS65": "EIF4EBP1",
    "4EBP1_pT37T46": "EIF4EBP1",
    "4EBP1_pT70": "EIF4EBP1",
    "53BP1": "TP53BP1",
    "ACC_pS79": None,
    "ACC1": "ACACA",
    "AKT": None,
    "AKT_pS473": None,
    "AKT_pT308": None,
    "AMPKALPHA": "PRKAA1",
    "AMPKALPHA_pT172": "PRKAA1",
    "AR": "AR",
    "ASNS": "ASNS",
    "ATM": "ATM",
    "BAK": "BAK1",
    "BAX": "BAX",
    "BCL2": "BCL2",
    "BCLXL": "BCL2L1",
    "BECLIN": "BECN1",
    "BETACATENIN": "CTNNB1",
    "BID": "BID",
    "BIM": "BCL2L11",
    "CJUN_pS73": "JUN",
    "CKIT": "KIT",
    "CMET": "MET",
    "CMET_pY1235": "MET",
    "CMYC": "MYC",
    "CRAF": "RAF1",
    "CRAF_pS338": "RAF1",
    "CASPASE7CLEAVEDD198": "CASP7",
    "CAVEOLIN1": "CAV1",
    "CD31": "PECAM1",
    "CD49B": "ITGA2",
    "CDK1": "CDK1",
    "CDK1_pY15": "CDK1",
    "CHK1": "CHEK1",
    "CHK1_pS296": "CHEK1",
    "CHK1_pS345": "CHEK1",
    "CHK2": "CHEK2",
    "CHK2_pT68": "CHEK2",
    "CIAP": "BIRC2",
    "CLAUDIN7": "CLDN7",
    "COLLAGENVI": "COL6A1",
    "CYCLINB1": "CCNB1",
    "CYCLIND1": "CCND1",
    "CYCLINE1": "CCNE1",
    "CYCLINE2": "CCNE2",
    "DJ1": "PARK7",
    "DVL3": "DVL3",
    "ECADHERIN": "CDH1",
    "EEF2": "EEF2",
    "EEF2K": "EEF2K",
    "EGFR": "EGFR",
    "EGFR_pY1068": "EGFR",
    "EGFR_pY1173": "EGFR",
    "EIF4E": "EIF4E",
    "EIF4G": "EIF4G1",
    "ERALPHA": "ESR1",
    "ERALPHA_pS118": "ESR1",
    "ERK2": "MAPK1",
    "FIBRONECTIN": "FN1",
    "FOXM1": "FOXM1",
    "FOXO3A": "FOXO3",
    "FOXO3A_pS318S321": "FOXO3",
    "GAB2": "GAB2",
    "GATA3": "GATA3",
    "GATA6": "GATA6",
    "GSK3ALPHABETA": None,
    "GSK3ALPHABETA_pS21S9": None,
    "GSK3_pS9": None,
    "HER2": "ERBB2",
    "HER2_pY1248": "ERBB2",
    "HER3": "ERBB3",
    "HER3_pY1289": "ERBB3",
    "HEREGULIN": "NRG1",
    "HSP70": "HSPA1A",
    "IGFBP2": "IGFBP2",
    "IGF1R_pY1135Y1136": "IGF1R",
    "INPP4B": "INPP4B",
    "IRS1": "IRS1",
    "JNK_pT183Y185": "MAPK8",
    "JNK2": "MAPK9",
    "KU80": "XRCC5",
    "LCK": "LCK",
    "LKB1": "STK11",
    "MAPK_pT202Y204": None,
    "MEK1": "MAP2K1",
    "MEK1_pS217S221": "MAP2K1",
    "MIG6": "ERRFI1",
    "MRE11": "MRE11A",
    "MTOR": "MTOR",
    "MTOR_pS2448": "MTOR",
    "MYH11": "MYH11",
    "MYOSINIIA": "MYH9",
    "MYOSINIIA_pS1943": "MYH9",
    "NCADHERIN": "CDH2",
    "NFKBP65_pS536": "RELA",
    "NF2": "NF2",
    "NOTCH1": "NOTCH1",
    "NRAS": "NRAS",
    "NDRG1_pT346": "NDRG1",
    "PCADHERIN": "CDH3",
    "P16INK4A": "CDKN2A",
    "P21": "CDKN1A",
    "P27": "CDKN1B",
    "P27_pT157": "CDKN1B",
    "P27_pT198": "CDKN1B",
    "P38MAPK": "MAPK14",
    "P38_pT180Y182": "MAPK14",
    "P53": "TP53",
    "P70S6K1": "RPS6KB1",
    "P70S6K_pT389": "RPS6KB1",
    "P90RSK": "RPS6KA1",
    "P90RSK_pT359S363": "RPS6KA1",
    "PAI1": "SERPINE1",
    "PAXILLIN": "PXN",
    "PCNA": "PCNA",
    "PDCD4": "PDCD4",
    "PDK1": "PDPK1",
    "PDK1_pS241": "PDPK1",
    "PDL1": "CD274",
    "PEA15": "PEA15",
    "PEA15_pS116": "PEA15",
    "PI3KP110ALPHA": "PIK3CA",
    "PI3KP85": "PIK3R1",
    "PKCALPHA": "PRKCA",
    "PKCALPHA_pS657": "PRKCA",
    "PKCDELTA_pS664": "PRKCD",
    "PKCPANBETAII_pS660": None,
    "PR": "PGR",
    "PRAS40_pT246": "AKT1S1",
    "PRDX1": "PRDX1",
    "PTEN": "PTEN",
    "RAB11": None,
    "RAB25": "RAB25",
    "RAD50": "RAD50",
    "RAD51": "RAD51",
    "RAPTOR": "RPTOR",
    "RB": "RB1",
    "RB_pS807S811": "RB1",
    "RBM15": "RBM15",
    "RICTOR": "RICTOR",
    "RICTOR_pT1135": "RICTOR",
    "S6": "RPS6",
    "S6_pS235S236": "RPS6",
    "S6_pS240S244": "RPS6",
    "SCD1": "SCD",
    "SF2": "SRSF1",
    "SHC_pY317": "SHC1",
    "SHP2_pY542": "PTPN11",
    "SMAD1": "SMAD1",
    "SMAD3": "SMAD3",
    "SMAD4": "SMAD4",
    "SRC": "SRC",
    "SRC_pY416": "SRC",
    "SRC_pY527": "SRC",
    "STAT3_pY705": "STAT3",
    "STAT5ALPHA": "STAT5A",
    "STATHMIN": "STMN1",
    "SYK": "SYK",
    "TAZ": "WWTR1",
    "TFRC": "TFRC",
    "TIGAR": "TIGAR",
    "TRANSGLUTAMINASE": "TGM2",
    "TSC1": "TSC1",
    "TUBERIN": "TSC2",
    "TUBERIN_pT1462": "TSC2",
    "VEGFR2": "KDR",
    "XRCC1": "XRCC1",
    "XBP1": "XBP1",
    "YAP": "YAP1",
    "YAP_pS127": "YAP1",
    "YB1": "YBX1",
    "YB1_pS102": "YBX1",
    "ARAF": "ARAF",
    "ARAF_pS299": "ARAF",
    "ANNEXINVII": "ANXA7",
    "ANNEXIN1": "ANXA1",
    "ARID1A": "ARID1A",
    "BRAF": "BRAF",
    "BRAF_pS445": "BRAF",
    "BAD_pS112": "BAD",
    "BAP1C4": "BAP1",
    "BRCA2": "BRCA2",
    "CD20": "MS4A1",
    "ETS1": "ETS1",
    "FASN": "FASN",
    "G6PD": "G6PD",
    "GAPDH": "GAPDH",
    "EPPK1": "EPPK1",
    "ACETYLATUBULINLYS40": "TUBA1B",
    "P62LCKLIGAND": "SQSTM1",
    "1433BETA": "YWHAB",
    "1433ZETA": "YWHAZ",
    "ACVRL1": "ACVRL1",
    "DIRAS3": "DIRAS3",
    "PREX1": "PREX1",
    "ENY2": "ENY2",
    "GCN5L2": "KAT2A",
    "ADAR1": "ADAR",
    "JAB1": "COPS5",
    "CASPASE8": "CASP8",
    "CASPASE3": "CASP3",
    "ERCC1": "ERCC1",
    "ERCC5": "ERCC5",
    "MSH2": "MSH2",
    "MSH6": "MSH6",
    "PARPCLEAVED": "PARP1",
    "SETD2": "SETD2",
    "SMAC": "DIABLO",
    "SNAIL": "SNAI1",
    "AXL": "AXL",
    "SLC1A5": "SLC1A5",
    "BRD4": "BRD4",
    "BCL2A1": "BCL2A1",
    "CABL": "ABL1",
    "CD26": "DPP4",
    "COG3": "COG3",
    "DUSP4": "DUSP4",
    "IRF1": "IRF1",
    "JAK2": "JAK2",
}


class PanelDataset(Dataset):
    def __init__(self, values: np.ndarray, mask: np.ndarray, y: np.ndarray):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        values = self.values[idx]
        return {
            "values": values,
            "mask": self.mask[idx],
            "label": self.y[idx],
            "CNV": values[:, 0],
            "mRNA": values[:, 1],
            "Proteomics": values[:, 2],
        }


def read_omic(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def read_labels(path: Path) -> np.ndarray:
    return pd.read_csv(path).iloc[:, 0].to_numpy()


def load_alias_map(cnv_symbols: set[str]) -> dict[str, str]:
    """Map a symbol or synonym onto the symbol used in the CNV matrix."""
    import gzip

    alias: dict[str, str] = {}
    if not GENE_INFO.is_file():
        return alias
    with gzip.open(GENE_INFO, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        sym_i = header.index("Symbol")
        syn_i = header.index("Synonyms")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            names = [parts[sym_i]]
            if parts[syn_i] not in ("", "-"):
                names.extend(parts[syn_i].split("|"))
            hit = next((name for name in names if name in cnv_symbols), None)
            if hit is None:
                continue
            for name in names:
                alias.setdefault(name, hit)
    return alias


def resolve_antibodies(antibodies: list[str], cnv_symbols: set[str], alias: dict[str, str]):
    grouped: dict[str, list[str]] = {}
    skipped_multi = []
    unresolved = []
    unknown = []
    for antibody in antibodies:
        if antibody not in ANTIBODY_TO_GENE:
            unknown.append(antibody)
            continue
        symbol = ANTIBODY_TO_GENE[antibody]
        if symbol is None:
            skipped_multi.append(antibody)
            continue
        resolved = symbol if symbol in cnv_symbols else alias.get(symbol)
        if resolved is None:
            unresolved.append((antibody, symbol))
            continue
        grouped.setdefault(resolved, []).append(antibody)
    return grouped, skipped_multi, unresolved, unknown


def scale_pair(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train).astype(np.float32)
    val_s = scaler.transform(val).astype(np.float32)
    return train_s, val_s


def aggregate_protein(train_df: pd.DataFrame, val_df: pd.DataFrame, grouped: dict[str, list[str]]):
    """Z-score each antibody on the training split, then average antibodies of one gene."""
    antibodies = [ab for group in grouped.values() for ab in group]
    train_s, val_s = scale_pair(train_df.loc[antibodies].T.to_numpy(), val_df.loc[antibodies].T.to_numpy())
    train_ab = pd.DataFrame(train_s, columns=antibodies)
    val_ab = pd.DataFrame(val_s, columns=antibodies)
    genes = list(grouped)
    train_g = np.column_stack([train_ab[grouped[gene]].mean(axis=1).to_numpy() for gene in genes])
    val_g = np.column_stack([val_ab[grouped[gene]].mean(axis=1).to_numpy() for gene in genes])
    train_g, val_g = scale_pair(train_g, val_g)
    return (
        pd.DataFrame(train_g, columns=genes),
        pd.DataFrame(val_g, columns=genes),
    )


def build_split(fold_dir: Path, top_k: int):
    cnv_tr = read_omic(fold_dir / "BRCA_CNV_1_channel_train.csv")
    cnv_va = read_omic(fold_dir / "BRCA_CNV_1_channel_val.csv")
    rna_tr = read_omic(fold_dir / "BRCA_mRNA_train.csv")
    rna_va = read_omic(fold_dir / "BRCA_mRNA_val.csv")
    prot_tr = read_omic(fold_dir / "BRCA_proteomics_train.csv")
    prot_va = read_omic(fold_dir / "BRCA_proteomics_val.csv")
    y_tr = read_labels(fold_dir / "BRCA_label_num_train.csv")
    y_va = read_labels(fold_dir / "BRCA_label_num_val.csv")

    if cnv_tr.shape[1] != len(y_tr) or cnv_va.shape[1] != len(y_va):
        raise RuntimeError(f"Sample counts do not match labels in {fold_dir}")

    cnv_symbols = set(cnv_tr.index)
    alias = load_alias_map(cnv_symbols)
    grouped, skipped_multi, unresolved, unknown = resolve_antibodies(list(prot_tr.index), cnv_symbols, alias)
    prot_train, prot_val = aggregate_protein(prot_tr, prot_va, grouped)

    rna_symbols = set(rna_tr.index)
    common = [gene for gene in cnv_tr.index if gene in rna_symbols]
    protein_genes = set(prot_train.columns) & set(common)
    aligned = [gene for gene in common if gene in protein_genes]
    if len(aligned) > top_k:
        keep = set(rna_tr.loc[aligned].var(axis=1).nlargest(top_k).index)
        panel = [gene for gene in aligned if gene in keep]
    else:
        panel = aligned

    # Scale each selected gene on the training patients only.
    cnv_tr_s, cnv_va_s = scale_pair(cnv_tr.loc[panel].T.to_numpy(), cnv_va.loc[panel].T.to_numpy())
    rna_tr_s, rna_va_s = scale_pair(rna_tr.loc[panel].T.to_numpy(), rna_va.loc[panel].T.to_numpy())
    protein_mask = np.array([gene in protein_genes for gene in panel], dtype=np.float32)

    def stack(cnv_s, rna_s, prot_df):
        protein = np.zeros((cnv_s.shape[0], len(panel)), dtype=np.float32)
        for col, gene in enumerate(panel):
            if gene in prot_df.columns:
                protein[:, col] = prot_df[gene].to_numpy(dtype=np.float32)
        values = np.stack([cnv_s, rna_s, protein], axis=-1)
        mask = np.broadcast_to(protein_mask, (cnv_s.shape[0], len(panel))).copy()
        return values, mask

    train_values, train_mask = stack(cnv_tr_s, rna_tr_s, prot_train)
    val_values, val_mask = stack(cnv_va_s, rna_va_s, prot_val)
    meta = {
        "n_panel": len(panel),
        "n_protein_genes": int(protein_mask.sum()),
        "n_top_only": int(sum(gene not in protein_genes for gene in panel)),
        "skipped_multi": skipped_multi,
        "unresolved": unresolved,
        "unknown_antibodies": unknown,
        "has_erbb2": "ERBB2" in panel,
        "has_esr1": "ESR1" in panel,
        "has_tp53": "TP53" in panel,
    }
    return train_values, train_mask, y_tr, val_values, val_mask, y_va, meta


def make_loader(values, mask, y, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        PanelDataset(values, mask, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


VARIANTS = {
    "A_omic_vector": "OSCAR on the panel: three omic vectors",
    "B_gene_genomic": "Gene tokens, genomic order, no within-gene sequence",
    "C_gene_dogma_genomic": "Within-gene CNV to mRNA to protein, genes in genomic order",
    "D_gene_dogma_shuffled": "Same as C with a fixed shuffle of the gene sequence",
}


def build_model(name, n_genes, n_classes, seed, lr, weight_decay, dropout):
    rng = np.random.default_rng(seed)
    shuffle = rng.permutation(n_genes).tolist()
    identity = list(range(n_genes))
    shared = dict(
        n_genes=n_genes,
        n_classes=n_classes,
        d_model=96,
        n_heads=4,
        n_layers=2,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
    )
    if name == "A_omic_vector":
        model = SelfAttentionFusionModel(
            input_dims={"CNV": n_genes, "mRNA": n_genes, "Proteomics": n_genes},
            num_classes=n_classes,
            latent_dim=64,
            encoder_hidden_dim=128,
            fusion_hidden=64,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            n_heads=4,
            n_layers=2,
            fusion_type="attention",
            use_cross_attention=True,
            cross_attn_hidden=64,
            use_original_for_proteomics=True,
            use_gated_fusion=True,
            gated_fusion_hidden=128,
        )
    elif name == "B_gene_genomic":
        model = GeneSequenceModel(perm=identity, within=False, **shared)
    elif name == "C_gene_dogma_genomic":
        model = GeneSequenceModel(perm=identity, within=True, **shared)
    elif name == "D_gene_dogma_shuffled":
        model = GeneSequenceModel(perm=shuffle, within=True, **shared)
    else:
        raise ValueError(name)
    return model


def fit_variant(model, train_loader, val_loader, epochs, patience):
    tracker = ValAccAtBestF1Callback()
    early = pl.callbacks.EarlyStopping(monitor="val_f1", mode="max", patience=patience, min_delta=0.001)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[tracker, early],
        deterministic=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)
    last = trainer.callback_metrics
    train_acc = last.get("train_acc")
    if train_acc is not None:
        train_acc = round(float(train_acc.detach().cpu()), 4)
    return {
        "f1": None if tracker.best_val_f1 is None else round(tracker.best_val_f1, 4),
        "accuracy": None if tracker.val_acc_at_best_f1 is None else round(tracker.val_acc_at_best_f1, 4),
        "train_acc_last": train_acc,
        "epochs_ran": int(trainer.current_epoch),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    frame = pd.DataFrame(rows)
    for name, group in frame.groupby("variant", sort=False):
        out[name] = {
            "f1_mean": round(float(group["f1"].mean()), 4),
            "f1_std": round(float(group["f1"].std(ddof=0)), 4),
            "accuracy_mean": round(float(group["accuracy"].mean()), 4),
            "accuracy_std": round(float(group["accuracy"].std(ddof=0)), 4),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.folds = 1
        args.epochs = 2
        args.patience = 2

    cfg = yaml.safe_load((ROOT / "configs" / "config_self_attn_cross_attn_brca.yaml").read_text(encoding="utf-8"))
    n_classes = int(cfg["model"]["num_classes"])
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    dropout = float(cfg["model"]["dropout"])

    rows = []
    panel_meta = []
    for fold in range(args.folds):
        fold_dir = FOLD_ROOT / f"fold_{fold + 1}"
        print(f"\n=== fold {fold + 1}/{args.folds} ===", flush=True)
        train_v, train_m, y_tr, val_v, val_m, y_va, meta = build_split(fold_dir, args.top_k)
        meta["fold"] = fold + 1
        panel_meta.append(meta)
        print(
            f"panel={meta['n_panel']} protein_genes={meta['n_protein_genes']} "
            f"unresolved={len(meta['unresolved'])} unknown={meta['unknown_antibodies']}",
            flush=True,
        )
        if meta["unknown_antibodies"] or meta["unresolved"]:
            print("unresolved", meta["unresolved"], "unknown", meta["unknown_antibodies"], flush=True)

        seed = args.seed + fold
        train_loader = make_loader(train_v, train_m, y_tr, args.batch_size, True, seed)
        val_loader = make_loader(val_v, val_m, y_va, args.batch_size, False, seed)
        # Each variant is constructed from the same seed, so initialization
        # noise is not part of the comparison.
        for name, description in VARIANTS.items():
            pl.seed_everything(seed, workers=True)
            model = build_model(name, meta["n_panel"], n_classes, seed, lr, weight_decay, dropout)
            print(f"  training {name}: {description}", flush=True)
            metrics = fit_variant(model, train_loader, val_loader, args.epochs, args.patience)
            print(
                f"  {name} f1={metrics['f1']} acc={metrics['accuracy']} "
                f"train_acc_last={metrics['train_acc_last']} "
                f"epochs={metrics['epochs_ran']} params={metrics['n_params']}",
                flush=True,
            )
            rows.append({"fold": fold + 1, "variant": name, "description": description, **metrics})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    averages = summarize(rows)
    payload = {
        "cohort": "BRCA",
        "panel_rule": "genes with CNV, mRNA, and a mapped RPPA antibody, in CNV genomic order",
        "top_k_cap": args.top_k,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "panel": panel_meta,
        "per_fold": rows,
        "averages": averages,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / ("smoke.json" if args.smoke else "results.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nAverages", flush=True)
    for name, stats in averages.items():
        print(
            f"{name}: F1 {stats['f1_mean']:.4f} +/- {stats['f1_std']:.4f}  "
            f"acc {stats['accuracy_mean']:.4f} +/- {stats['accuracy_std']:.4f}",
            flush=True,
        )
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
