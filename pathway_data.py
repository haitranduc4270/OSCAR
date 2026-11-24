"""Build the BRCA pathway-token cohort and the fixed seed-42 split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pathway_sets import (
    ERBB2_GENES,
    ERBB2_PATHWAY,
    PATHFORMER_PROTEIN_READS,
    assign_protein_programs,
    is_phospho,
    load_hallmark_gmt,
    load_pathformer_pathways,
)


def _read_split_matrix(fold_dir: Path, stem: str) -> pd.DataFrame:
    """Genes by samples, train columns followed by validation columns."""
    train = pd.read_csv(fold_dir / f"{stem}_train.csv", index_col=0)
    val = pd.read_csv(fold_dir / f"{stem}_val.csv", index_col=0)
    val = val.reindex(train.index)
    if val.isna().any().any():
        raise ValueError(f"Train and val features do not align for {stem}")
    return pd.concat([train, val], axis=1)


def _pad_indices(groups: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(group) for group in groups), default=0)
    index = torch.zeros(len(groups), width, dtype=torch.long)
    mask = torch.zeros(len(groups), width, dtype=torch.bool)
    for row, group in enumerate(groups):
        if not group:
            continue
        index[row, : len(group)] = torch.tensor(group, dtype=torch.long)
        mask[row, : len(group)] = True
    return index, mask


def _load_pathway_sets(data_cfg: dict) -> tuple[dict[str, list[str]], dict[str, tuple[str, ...]] | None, str]:
    if data_cfg.get("pathformer_pathway_table"):
        sets = load_pathformer_pathways(data_cfg["pathformer_pathway_table"])
        return sets, PATHFORMER_PROTEIN_READS, "pathformer"
    if data_cfg.get("hallmark_gmt"):
        return load_hallmark_gmt(data_cfg["hallmark_gmt"]), None, "hallmark"
    raise ValueError("data config needs hallmark_gmt or pathformer_pathway_table")


def build_cohort(cfg: dict) -> dict:
    data_cfg = cfg["data"]
    fold_dir = Path(data_cfg["pre_split_fold"])
    pathway_sets, protein_reads, pathway_source = _load_pathway_sets(data_cfg)

    mrna = _read_split_matrix(fold_dir, data_cfg["mrna_stem"])
    cnv = _read_split_matrix(fold_dir, data_cfg["cnv_stem"])
    protein = _read_split_matrix(fold_dir, data_cfg["protein_stem"])
    if list(mrna.columns) != list(cnv.columns) or list(mrna.columns) != list(protein.columns):
        raise ValueError("Sample columns differ across CNV, mRNA and proteomics")

    y_train = pd.read_csv(fold_dir / f"{data_cfg['label_stem']}_train.csv").iloc[:, 0].to_numpy()
    y_val = pd.read_csv(fold_dir / f"{data_cfg['label_stem']}_val.csv").iloc[:, 0].to_numpy()
    y = np.concatenate([y_train, y_val]).astype(np.int64)
    if len(y) != mrna.shape[1]:
        raise ValueError(f"{len(y)} labels for {mrna.shape[1]} samples")

    measured = set(mrna.index) & set(cnv.index)
    union = set().union(*pathway_sets.values())
    for gene in ERBB2_GENES:
        if gene not in measured:
            raise ValueError(f"{gene} is missing from CNV or mRNA")
    genes = sorted((union & measured) | (set(ERBB2_GENES) & measured))
    excluded = sorted(union - set(genes))

    pathway_names = list(pathway_sets) + [ERBB2_PATHWAY]
    gene_index = {gene: i for i, gene in enumerate(genes)}
    membership = np.zeros((len(pathway_names), len(genes)), dtype=np.float32)
    members_by_pathway = {name: genes_in for name, genes_in in pathway_sets.items()}
    members_by_pathway[ERBB2_PATHWAY] = list(ERBB2_GENES)
    for row, name in enumerate(pathway_names):
        for gene in members_by_pathway[name]:
            column = gene_index.get(gene)
            if column is not None:
                membership[row, column] = 1.0
        if membership[row].sum() == 0:
            raise ValueError(f"Pathway {name} has no measured gene")

    programs = assign_protein_programs(
        [str(name) for name in protein.index],
        reads_by_program=protein_reads,
    )
    antibody_index = {name: i for i, name in enumerate(protein.index.astype(str))}
    total_groups: list[list[int]] = []
    phospho_groups: list[list[int]] = []
    program_names: list[str] = []
    program_reads: list[tuple[str, ...]] = []
    for name, members, reads in programs:
        program_names.append(name)
        program_reads.append(reads)
        total_groups.append([antibody_index[a] for a in members if not is_phospho(a)])
        phospho_groups.append([antibody_index[a] for a in members if is_phospho(a)])

    pathway_row = {name: i for i, name in enumerate(pathway_names)}
    n_pathways = len(pathway_names)
    key_groups: list[list[int]] = []
    for reads in program_reads:
        keys: list[int] = []
        for pathway in reads:
            if pathway not in pathway_row:
                raise ValueError(f"Protein program reads an unknown pathway: {pathway}")
            row = pathway_row[pathway]
            keys.append(row)
            keys.append(n_pathways + row)
        key_groups.append(keys)

    total_index, total_mask = _pad_indices(total_groups)
    phospho_index, phospho_mask = _pad_indices(phospho_groups)
    key_index, key_mask = _pad_indices(key_groups)

    seed = int(data_cfg["seed"])
    indices = np.arange(len(y))
    dev_idx, test_idx = train_test_split(
        indices,
        test_size=float(data_cfg["test_size"]),
        stratify=y,
        random_state=seed,
    )
    train_idx, val_idx = train_test_split(
        dev_idx,
        test_size=float(data_cfg["val_fraction_of_remaining"]),
        stratify=y[dev_idx],
        random_state=seed,
    )
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)

    def counts(idx: np.ndarray) -> list[int]:
        return np.bincount(y[idx], minlength=int(y.max()) + 1).tolist()

    return {
        "mrna": mrna.loc[genes].T.to_numpy(dtype=np.float64),
        "cnv": cnv.loc[genes].T.to_numpy(dtype=np.float64),
        "protein": protein.T.to_numpy(dtype=np.float64),
        "y": y,
        "genes": genes,
        "excluded_genes": excluded,
        "pathway_source": pathway_source,
        "hallmark_union": len(union),
        "pathway_names": pathway_names,
        "membership": torch.from_numpy(membership),
        "program_names": program_names,
        "program_reads": program_reads,
        "program_sizes": {name: len(members) for name, members, _reads in programs},
        "total_index": total_index,
        "total_mask": total_mask,
        "phospho_index": phospho_index,
        "phospho_mask": phospho_mask,
        "key_index": key_index,
        "key_mask": key_mask,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "split_counts": {
            "n_total": int(len(y)),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "train_labels": counts(train_idx),
            "val_labels": counts(val_idx),
            "test_labels": counts(test_idx),
        },
    }


def scale_matrices(
    cohort: dict,
    fit_idx: np.ndarray,
    apply_idx: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Z-score each feature on ``fit_idx`` and transform the requested splits."""
    scaled: dict[str, dict[str, np.ndarray]] = {}
    for name in ("mrna", "cnv", "protein"):
        scaler = StandardScaler()
        scaler.fit(cohort[name][fit_idx])
        scaled[name] = {
            split: scaler.transform(cohort[name][idx]).astype(np.float32)
            for split, idx in apply_idx.items()
        }
    return scaled
