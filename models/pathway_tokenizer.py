"""Pathway-guided projection and tokenization modules for TokenOSCAR.

Provides two sparse biological alternatives to OmicToTokens:
1. PathwayBottleneckToTokens: replaces the dense first layer Linear(D, d_mid)
   with an edge-indexed sparse gene->pathway layer (plus optional unmapped-gene
   residual branch), then projects to (B, T, M).
2. PathwayTokenEncoder: maps each biological pathway directly into one token of
   dimension M via K learned per-pathway gene projections, producing (B, P, M).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent

PATHWAY_FILES = {
    "hallmark": ROOT / "resources/msigdb/h.all.v7.4.symbols.gmt",
    "kegg": ROOT / "third_party/pathformer/reference/pathway_PathCNN.txt",
    "pathformer": ROOT / "third_party/pathformer/reference/Pathformer_pathway.txt",
}


def load_pathway_sets(source: str) -> dict[str, set[str]]:
    """Load pathway -> set of uppercase HGNC gene symbols from local files."""
    if source not in PATHWAY_FILES:
        raise ValueError(f"Unknown pathway source '{source}'. Expected one of {list(PATHWAY_FILES)}")
    path = PATHWAY_FILES[source]
    if not path.is_file():
        raise FileNotFoundError(f"Pathway file not found: {path}")

    pathways: dict[str, set[str]] = {}
    if source == "hallmark":
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            name = parts[0].strip()
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            if name and genes:
                pathways[name] = genes
    elif source == "kegg":
        df = pd.read_csv(path, sep="\t")
        for _, row in df.iterrows():
            name = str(row["ID"]).strip()
            genes = {g.strip().upper() for g in str(row["gene_list"]).split(",") if g.strip()}
            if name and genes:
                pathways[name] = genes
    elif source == "pathformer":
        df = pd.read_csv(path, sep="\t")
        for _, row in df.iterrows():
            name = str(row["name"]).strip()
            genes = {g.strip().upper() for g in str(row["gene"]).split(",") if g.strip()}
            if name and genes:
                pathways[name] = genes

    return dict(sorted(pathways.items()))


# Curated RPPA antibody base name (uppercased, stripped of hyphens/underscores/phospho/TCPA suffix)
# -> HGNC gene symbol(s). Multi-isoform antibodies map to their target paralogs so each
# phospho-site/antibody receives independent learned edge weights into its pathways.
RPPA_ALIAS_TO_GENES: dict[str, list[str]] = {
    "1433BETA": ["YWHAB"],
    "1433EPSILON": ["YWHAE"],
    "1433ZETA": ["YWHAZ"],
    "4EBP1": ["EIF4EBP1"],
    "53BP1": ["TP53BP1"],
    "ACC": ["ACACA", "ACACB"],
    "ACC1": ["ACACA"],
    "ACETYLATUBULIN": ["TUBA1B"],
    "ADAR1": ["ADAR"],
    "AKT": ["AKT1", "AKT2", "AKT3"],
    "AMPK": ["PRKAA1", "PRKAA2"],
    "AMPKALPHA": ["PRKAA1", "PRKAA2"],
    "ANNEXIN1": ["ANXA1"],
    "ANNEXINVII": ["ANXA7"],
    "ARAF": ["ARAF"],
    "BAK": ["BAK1"],
    "BAP1C4": ["BAP1"],
    "BCLXL": ["BCL2L1"],
    "BECLIN": ["BECN1"],
    "BETACATENIN": ["CTNNB1"],
    "BIM": ["BCL2L11"],
    "BRAF": ["BRAF"],
    "CABL": ["ABL1"],
    "CASPASE3": ["CASP3"],
    "CASPASE7": ["CASP7"],
    "CASPASE8": ["CASP8"],
    "CAVEOLIN1": ["CAV1"],
    "CD20": ["MS4A1"],
    "CD26": ["DPP4"],
    "CD31": ["PECAM1"],
    "CD49B": ["ITGA2"],
    "CHK1": ["CHEK1"],
    "CHK2": ["CHEK2"],
    "CIAP": ["BIRC2"],
    "CJUN": ["JUN"],
    "CKIT": ["KIT"],
    "CLAUDIN7": ["CLDN7"],
    "CMET": ["MET"],
    "CMYC": ["MYC"],
    "COLLAGENVI": ["COL6A1"],
    "CRAF": ["RAF1"],
    "CYCLINB1": ["CCNB1"],
    "CYCLIND1": ["CCND1"],
    "CYCLINE1": ["CCNE1"],
    "CYCLINE2": ["CCNE2"],
    "DJ1": ["PARK7"],
    "ECADHERIN": ["CDH1"],
    "EIF4G": ["EIF4G1"],
    "ERALPHA": ["ESR1"],
    "ERK2": ["MAPK1"],
    "FIBRONECTIN": ["FN1"],
    "FOXO3A": ["FOXO3"],
    "GCN5L2": ["KAT2A"],
    "GSK3": ["GSK3A", "GSK3B"],
    "GSK3ALPHABETA": ["GSK3A", "GSK3B"],
    "HER2": ["ERBB2"],
    "HER3": ["ERBB3"],
    "HEREGULIN": ["NRG1"],
    "HSP70": ["HSPA1A"],
    "JAB1": ["COPS5"],
    "JNK": ["MAPK8", "MAPK9", "MAPK10"],
    "JNK2": ["MAPK9"],
    "KU80": ["XRCC5"],
    "LKB1": ["STK11"],
    "MAPK": ["MAPK1", "MAPK3"],
    "MEK1": ["MAP2K1"],
    "MIG6": ["ERRFI1"],
    "MRE11": ["MRE11A", "MRE11"],
    "MRE11A": ["MRE11", "MRE11A"],
    "MYOSINIIA": ["MYH9"],
    "NCADHERIN": ["CDH2"],
    "NFKBP65": ["RELA"],
    "P16INK4A": ["CDKN2A"],
    "P21": ["CDKN1A"],
    "P27": ["CDKN1B"],
    "P38": ["MAPK14"],
    "P38MAPK": ["MAPK14"],
    "P53": ["TP53"],
    "P62LCKLIGAND": ["SQSTM1"],
    "P70S6K": ["RPS6KB1"],
    "P70S6K1": ["RPS6KB1"],
    "P90RSK": ["RPS6KA1"],
    "PAI1": ["SERPINE1"],
    "PARP": ["PARP1"],
    "PAXILLIN": ["PXN"],
    "PCADHERIN": ["CDH3"],
    "PDL1": ["CD274"],
    "PI3KP110ALPHA": ["PIK3CA"],
    "PI3KP85": ["PIK3R1"],
    "PKCALPHA": ["PRKCA"],
    "PKCDELTA": ["PRKCD"],
    "PKCPANBETAII": ["PRKCB"],
    "PR": ["PGR"],
    "PRAS40": ["AKT1S1"],
    "RAB11": ["RAB11A", "RAB11B"],
    "RAPTOR": ["RPTOR"],
    "RB": ["RB1"],
    "S6": ["RPS6"],
    "SCD1": ["SCD"],
    "SF2": ["SRSF1"],
    "SHC": ["SHC1"],
    "SHP2": ["PTPN11"],
    "SMAC": ["DIABLO"],
    "SNAIL": ["SNAI1"],
    "STAT5ALPHA": ["STAT5A"],
    "STATHMIN": ["STMN1"],
    "TAZ": ["WWTR1"],
    "TRANSGLUTAMINASE": ["TGM2"],
    "TTF1": ["NKX2-1"],
    "TUBERIN": ["TSC2"],
    "VEGFR2": ["KDR"],
    "YAP": ["YAP1"],
    "YB1": ["YBX1"],
}


def rppa_feature_to_genes(feature_name: str) -> list[str]:
    """Map an RPPA antibody/phospho-protein feature name to HGNC gene symbol(s)."""
    s = str(feature_name).strip()
    # Keep direct HGNC symbols like NKX2-1 intact before hyphen stripping
    upper_raw = s.upper()
    # Strip TCPA antibody validation suffixes: -R-V, -M-V, -R-C, -M-E, -G-C
    s = re.sub(r"-[RMG]-[VCE]$", "", s)
    # Strip phosphorylation / cleavage / acetylation site annotations
    s = re.sub(r"_p[STY]\d.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_?cleaved.*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"LYS\d+$", "", s, flags=re.IGNORECASE)
    # Keep true hyphenated HGNC symbols intact before hyphen stripping
    if s.upper() == "NKX2-1":
        return ["NKX2-1"]
    norm = re.sub(r"[-_]", "", s).upper()
    if norm in RPPA_ALIAS_TO_GENES:
        return RPPA_ALIAS_TO_GENES[norm]
    if upper_raw in RPPA_ALIAS_TO_GENES:
        return RPPA_ALIAS_TO_GENES[upper_raw]
    return [norm]


def build_pathway_spec(
    feature_names: Sequence[str],
    pathway_dict: dict[str, set[str]],
    feature_mapper: Callable[[str], Sequence[str]] | None = None,
) -> dict:
    """Build edge-indexed gene->pathway membership specification for one omic."""
    symbol_to_indices: dict[str, list[int]] = {}
    for idx, feat in enumerate(feature_names):
        if feature_mapper is not None:
            syms = [str(g).strip().upper() for g in feature_mapper(str(feat)) if str(g).strip()]
        else:
            syms = [str(feat).strip().upper()]
        for sym in syms:
            symbol_to_indices.setdefault(sym, []).append(idx)

    pathway_names = sorted(pathway_dict.keys())
    gene_idx: list[int] = []
    pw_idx: list[int] = []
    pw_deg: list[int] = []
    mapped_genes: set[int] = set()
    active_pathways = 0

    for p_i, p_name in enumerate(pathway_names):
        genes = pathway_dict[p_name]
        matched_indices: list[int] = []
        for g in genes:
            if g in symbol_to_indices:
                matched_indices.extend(symbol_to_indices[g])
        matched_indices = sorted(set(matched_indices))
        deg = len(matched_indices)
        if deg > 0:
            active_pathways += 1
            for g_i in matched_indices:
                gene_idx.append(g_i)
                pw_idx.append(p_i)
                mapped_genes.add(g_i)
        pw_deg.append(max(deg, 1))

    n_features = len(feature_names)
    unmapped_idx = [i for i in range(n_features) if i not in mapped_genes]

    return {
        "pathway_names": pathway_names,
        "num_pathways": len(pathway_names),
        "gene_idx": gene_idx,
        "pw_idx": pw_idx,
        "pw_deg": pw_deg,
        "unmapped_idx": unmapped_idx,
        "coverage": {
            "n_features": n_features,
            "n_mapped": len(mapped_genes),
            "n_unmapped": len(unmapped_idx),
            "coverage_pct": round(100.0 * len(mapped_genes) / max(n_features, 1), 2),
            "n_edges": len(gene_idx),
            "num_pathways": len(pathway_names),
            "n_active_pathways": active_pathways,
        },
    }


class PathwayBottleneckToTokens(nn.Module):
    """Sparse gene->pathway bottleneck (B, D) -> (B, P [+ U]) -> (B, T, M)."""

    def __init__(
        self,
        spec: dict,
        num_tokens: int = 256,
        token_dim: int = 32,
        dropout: float = 0.4,
        unmapped_dim: int = 0,
        token_layernorm: bool = False,
    ):
        super().__init__()
        self.num_pathways = int(spec["num_pathways"])
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)

        gene_idx = torch.tensor(spec["gene_idx"], dtype=torch.long)
        pw_idx = torch.tensor(spec["pw_idx"], dtype=torch.long)
        pw_deg = torch.tensor(spec["pw_deg"], dtype=torch.float32)

        self.register_buffer("gene_idx", gene_idx)
        self.register_buffer("pw_idx", pw_idx)

        n_edges = len(spec["gene_idx"])
        self.edge_weight = nn.Parameter(torch.empty(n_edges))
        self.pw_bias = nn.Parameter(torch.zeros(self.num_pathways))
        with torch.no_grad():
            bounds = 1.0 / torch.sqrt(pw_deg[pw_idx])
            self.edge_weight.uniform_(-1.0, 1.0).mul_(bounds)

        unmapped_idx = spec["unmapped_idx"]
        self.use_unmapped = bool(unmapped_dim > 0 and len(unmapped_idx) > 0)
        if self.use_unmapped:
            self.register_buffer("unmapped_idx", torch.tensor(unmapped_idx, dtype=torch.long))
            self.unmapped_proj = nn.Linear(len(unmapped_idx), int(unmapped_dim))
            bottleneck_dim = self.num_pathways + int(unmapped_dim)
        else:
            self.unmapped_proj = None
            bottleneck_dim = self.num_pathways

        self.expand = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, self.num_tokens * self.token_dim),
        )
        self.post_norm = nn.LayerNorm(self.token_dim) if token_layernorm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        contrib = x.index_select(1, self.gene_idx) * self.edge_weight.unsqueeze(0)
        pw = x.new_zeros((B, self.num_pathways))
        pw.index_add_(1, self.pw_idx, contrib)
        pw = pw + self.pw_bias.unsqueeze(0)

        if self.use_unmapped and self.unmapped_proj is not None:
            u = self.unmapped_proj(x.index_select(1, self.unmapped_idx))
            h = torch.cat([pw, u], dim=-1)
        else:
            h = pw

        out = self.expand(h).view(B, self.num_tokens, self.token_dim)
        return self.post_norm(out) if self.post_norm is not None else out


class PathwayTokenEncoder(nn.Module):
    """Direct pathway-as-tokens encoder: (B, D) -> (B, P, M) with strictly local pathway grouping."""

    def __init__(
        self,
        spec: dict,
        token_dim: int = 32,
        proj_per_pathway: int = 8,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.num_pathways = int(spec["num_pathways"])
        self.token_dim = int(token_dim)
        self.proj_per_pathway = int(proj_per_pathway)

        gene_idx = torch.tensor(spec["gene_idx"], dtype=torch.long)
        pw_idx = torch.tensor(spec["pw_idx"], dtype=torch.long)
        pw_deg = torch.tensor(spec["pw_deg"], dtype=torch.float32)

        self.register_buffer("gene_idx", gene_idx)
        self.register_buffer("pw_idx", pw_idx)

        n_edges = len(spec["gene_idx"])
        self.edge_weight = nn.Parameter(torch.empty(n_edges, self.proj_per_pathway))
        self.pw_bias = nn.Parameter(torch.zeros(self.num_pathways, self.proj_per_pathway))
        with torch.no_grad():
            bounds = (1.0 / torch.sqrt(pw_deg[pw_idx])).unsqueeze(-1)
            self.edge_weight.uniform_(-1.0, 1.0).mul_(bounds)

        self.token_mlp = nn.Sequential(
            nn.LayerNorm(self.proj_per_pathway),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.proj_per_pathway, self.token_dim),
        )
        self.pathway_embed = nn.Parameter(torch.randn(1, self.num_pathways, self.token_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # (B, E, 1) * (1, E, K) -> (B, E, K)
        contrib = x.index_select(1, self.gene_idx).unsqueeze(-1) * self.edge_weight.unsqueeze(0)
        pw_feats = x.new_zeros((B, self.num_pathways, self.proj_per_pathway))
        pw_feats.index_add_(1, self.pw_idx, contrib)
        pw_feats = pw_feats + self.pw_bias.unsqueeze(0)
        return self.token_mlp(pw_feats) + self.pathway_embed
