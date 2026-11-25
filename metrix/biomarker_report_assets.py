"""
Report-ready tables and figures from biomarker steps 1–3.

Paper subtype names follow the cohort counts in report.tex:
LumA 248, LumB 98, Basal 112, HER2 53 (class indices 0, 1, 2, 3).
The current DE file already uses those names. The Enrichr tags from step 3
were written with HER2/Basal swapped relative to that table; this script retags them.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from biomarker_step2_de import load_full_cohort

OUT = Path("attention_plots/biomarker_report_assets")
DE_CSV = Path("attention_plots/biomarker_step2_de/consensus_with_de.csv")
ENR_CSV = Path("attention_plots/biomarker_step3_enrichment/enrichment_all.csv")
CONFIG = Path("configs/config_self_attn_cross_attn_brca.yaml")

# Paper Table dataset_summary counts.
COUNT_TO_PAPER = {248: "LumA", 98: "LumB", 112: "Basal", 53: "HER2"}
PAPER_ORDER = ["LumA", "LumB", "Basal", "HER2"]
# NPG-like palette shared with the consensus lollipop figures.
OMIC_COLORS = {"mRNA": "#3C5488", "CNV": "#E64B35", "Proteomics": "#00A087"}
OMIC_ORDER = ["mRNA", "CNV", "Proteomics"]
DISPLAY_NAME = {
    "LumA": "Luminal A",
    "LumB": "Luminal B",
    "Basal": "Basal-like",
    "HER2": "HER2-enriched",
}
LIB_COLOR = {
    "MSigDB_Hallmark_2020": "#E64B35",
    "GO_Biological_Process_2025": "#3C5488",
}


def apply_mpl_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 0.7,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def style_ax(ax, grid: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.0, width=0.6, color="#444444")
    ax.grid(axis=grid, linestyle=":", color="#D5D5D5", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def italic_label(name: str) -> str:
    safe = str(name).replace("_", r"\_")
    return rf"$\it{{{safe}}}$"

MARKERS = [
    "ESR1",
    "PGR",
    "FOXA1",
    "GATA3",
    "ERBB2",
    "GRB7",
    "STARD3",
    "MIEN1",
    "PGAP3",
    "KRT5",
    "KRT14",
    "KRT17",
    "EGFR",
    "FOXC1",
    "MKI67",
    "AURKA",
    "BCL2",
]


def short_term(term: str) -> str:
    term = re.sub(r"\s*\(GO:\d+\)", "", str(term))
    if len(term) > 62:
        return term[:59] + "..."
    return term


def audit_labels(y: np.ndarray, de_raw: pd.DataFrame) -> pd.DataFrame:
    """Paper names follow cohort counts, not the string stored in an older CSV."""
    rows = []
    for idx in sorted(np.unique(y)):
        idx = int(idx)
        n = int((y == idx).sum())
        csv_names = de_raw.loc[de_raw["class_idx"] == idx, "class_name"]
        csv_name = csv_names.mode().iloc[0] if len(csv_names) else ""
        rows.append(
            {
                "class_idx": idx,
                "n": n,
                "de_csv_name": csv_name,
                "paper_name": COUNT_TO_PAPER.get(n, f"n={n}"),
            }
        )
    return pd.DataFrame(rows)


def marker_means(y: np.ndarray, feature_names: dict, X: dict, audit: pd.DataFrame) -> pd.DataFrame:
    name_of = dict(zip(audit["class_idx"], audit["paper_name"]))
    rows = []
    for omic in ("mRNA", "CNV", "Proteomics"):
        names = feature_names[omic]
        index = {g: i for i, g in enumerate(names)}
        wanted = MARKERS if omic != "Proteomics" else ["ERBB2", "HER2", "GATA3", "BCL2", "CKIT", "P53"]
        for gene in wanted:
            if gene not in index:
                continue
            col = X[omic][:, index[gene]]
            for idx in sorted(name_of):
                mask = y == idx
                rows.append(
                    {
                        "omic": omic,
                        "gene": gene,
                        "class_idx": int(idx),
                        "paper_name": name_of[idx],
                        "mean": float(np.nanmean(col[mask])),
                        "n": int(mask.sum()),
                    }
                )
    return pd.DataFrame(rows)


def apply_paper_names(df: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Rename by class index. The DE file's class_name already matches these counts."""
    mapping = dict(zip(audit["class_idx"], audit["paper_name"]))
    out = df.copy()
    out["de_csv_name"] = out["class_name"]
    out["class_name"] = out["class_idx"].map(lambda i: mapping.get(int(i), str(i)))
    return out


def retag_enrichment(enr: pd.DataFrame) -> pd.DataFrame:
    """Step-3 tags used --class-names LumA,LumB,HER2,Basal.

    That string calls index 2 (n=112) HER2 and index 3 (n=53) Basal.
    Paper counts and the current DE file use the opposite pair.
    """
    swap = {"LumA": "LumA", "LumB": "LumB", "HER2": "Basal", "Basal": "HER2"}
    out = enr.copy()

    def fix(tag: str) -> str:
        for src, dst in swap.items():
            if tag.startswith(src + "_") or tag == src:
                return dst + tag[len(src) :]
        return tag

    out["tag_script"] = out["tag"]
    out["tag"] = out["tag"].map(fix)
    return out


def plot_marker_heatmap(means: pd.DataFrame, path: Path) -> None:
    sub = means[(means["omic"] == "mRNA") & (means["gene"].isin(MARKERS))].copy()
    mat = sub.pivot(index="gene", columns="paper_name", values="mean")
    mat = mat.reindex(index=[g for g in MARKERS if g in mat.index], columns=PAPER_ORDER)
    fig, ax = plt.subplots(figsize=(6.2, 7.2))
    vmax = np.nanmax(np.abs(mat.to_numpy()))
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.to_numpy()[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_title("Mean mRNA by subtype (centered scale)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cohens_d(de: pd.DataFrame, path: Path) -> None:
    apply_mpl_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.4))
    letters = ["A", "B", "C", "D"]
    for ax, name, letter in zip(axes.ravel(), PAPER_ORDER, letters):
        sub = de[de["class_name"] == name].sort_values("cohens_d")
        ypos = np.arange(len(sub))
        colors = [OMIC_COLORS.get(o, "#777777") for o in sub["omic"]]
        ax.axvline(0, color="#4A4A4A", lw=0.7, zorder=1)
        ax.hlines(ypos, 0, sub["cohens_d"], color=colors, lw=1.35, zorder=2)
        for y, row, c in zip(ypos, sub.itertuples(), colors):
            ax.scatter(
                row.cohens_d,
                y,
                s=36,
                c=c if row.de_pass else "white",
                edgecolors=c,
                linewidths=1.05,
                zorder=3,
            )
        ax.set_yticks(ypos)
        ax.set_yticklabels([italic_label(f) for f in sub["feature"]], fontsize=7.5)
        ax.set_title(f"{letter}    {DISPLAY_NAME[name]}", loc="left", fontweight="bold", pad=8)
        n_pass = int(sub["de_pass"].sum()) if len(sub) else 0
        ax.text(
            0.98,
            0.04,
            f"{n_pass}/{len(sub)} pass",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#555555",
        )
        style_ax(ax)
    for ax in axes[1, :]:
        ax.set_xlabel("Cohen's $d$ (subtype vs rest)")
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=OMIC_COLORS[o],
            markeredgecolor="white", markersize=7, label=o,
        )
        for o in OMIC_ORDER
    ]
    handles.append(
        plt.Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="white",
            markeredgecolor="#666666", markersize=7, label="Gradient only",
        )
    )
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_enrichment(enr: pd.DataFrame, path: Path) -> None:
    tag = "LumA_consensus"
    libs = ["MSigDB_Hallmark_2020", "GO_Biological_Process_2025"]
    sub = enr[(enr["tag"] == tag) & (enr["Gene_set"].isin(libs))].copy()
    sub = sub[sub["Adjusted P-value"] < 0.05]
    pieces = []
    for lib, n in (("MSigDB_Hallmark_2020", 6), ("GO_Biological_Process_2025", 8)):
        piece = sub[sub["Gene_set"] == lib].sort_values("Adjusted P-value").head(n)
        pieces.append(piece)
    plot_df = pd.concat(pieces, ignore_index=True)
    plot_df["label"] = plot_df["Term"].map(short_term)
    plot_df["neglog10"] = -np.log10(plot_df["Adjusted P-value"].clip(lower=1e-300))
    plot_df["n_overlap"] = plot_df["Overlap"].astype(str).str.split("/").str[0].astype(int)
    # Most significant term at the top, Hallmark block above GO.
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ypos = np.arange(len(plot_df))
    colors = [LIB_COLOR[g] for g in plot_df["Gene_set"]]
    sizes = 28 + 22 * plot_df["n_overlap"].to_numpy()
    ax.axvline(-np.log10(0.05), color="#B0B0B0", lw=0.7, ls="--", zorder=1)
    ax.hlines(ypos, 0, plot_df["neglog10"], color=colors, lw=1.15, zorder=2)
    ax.scatter(
        plot_df["neglog10"], ypos, s=sizes, c=colors, zorder=3,
        edgecolors="white", linewidths=0.5,
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(plot_df["label"], fontsize=8)
    ax.set_xlabel(r"$-\log_{10}$(adjusted $P$)")
    ax.set_xlim(0, float(plot_df["neglog10"].max()) * 1.08)
    style_ax(ax)
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=LIB_COLOR[k],
            markeredgecolor="white", markersize=7, label=lab,
        )
        for k, lab in (
            ("MSigDB_Hallmark_2020", "MSigDB Hallmark"),
            ("GO_Biological_Process_2025", "GO Biological Process"),
        )
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plot_df.iloc[::-1].to_csv(path.with_suffix(".csv"), index=False)


def concordance_table(de: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in de.groupby("class_name"):
        cnv = sub[sub["omic"] == "CNV"].set_index("feature")
        rna = sub[sub["omic"] == "mRNA"].set_index("feature")
        shared = sorted(set(cnv.index) & set(rna.index))
        for gene in shared:
            a, b = cnv.loc[gene], rna.loc[gene]
            if isinstance(a, pd.DataFrame):
                a = a.iloc[0]
            if isinstance(b, pd.DataFrame):
                b = b.iloc[0]
            rows.append(
                {
                    "class_name": name,
                    "gene": gene,
                    "d_cnv": float(a["cohens_d"]),
                    "d_mrna": float(b["cohens_d"]),
                    "q_cnv": float(a["q_value"]),
                    "q_mrna": float(b["q_value"]),
                    "pass_cnv": bool(a["de_pass"]),
                    "pass_mrna": bool(b["de_pass"]),
                    "same_sign": bool(np.sign(a["cohens_d"]) == np.sign(b["cohens_d"])),
                    "n_folds_cnv": int(a["n_folds"]),
                    "n_folds_mrna": int(b["n_folds"]),
                }
            )
    return pd.DataFrame(rows)


def plot_concordance(conc: pd.DataFrame, path: Path) -> None:
    sub = conc[conc["class_name"] == "HER2"].sort_values("d_mrna")
    if sub.empty:
        return
    apply_mpl_style()
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    ypos = np.arange(len(sub))
    for y, row in zip(ypos, sub.itertuples()):
        ax.plot([row.d_cnv, row.d_mrna], [y, y], color="#C8C8C8", lw=1.6, zorder=1, solid_capstyle="round")
    ax.scatter(
        sub["d_cnv"], ypos, s=52, c=OMIC_COLORS["CNV"], label="CNV",
        zorder=3, edgecolors="white", linewidths=0.6,
    )
    ax.scatter(
        sub["d_mrna"], ypos, s=52, c=OMIC_COLORS["mRNA"], label="mRNA",
        zorder=3, edgecolors="white", linewidths=0.6,
    )
    ax.axvline(0, color="#4A4A4A", lw=0.7, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([italic_label(g) for g in sub["gene"]], fontsize=9)
    ax.set_xlabel("Cohen's $d$ (HER2-enriched vs rest)")
    ax.set_xlim(left=-0.05)
    style_ax(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.62, 1.02), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    header = "| " + " | ".join(map(str, show.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in show.columns) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in show.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def write_numbers(path: Path, audit: pd.DataFrame, de: pd.DataFrame, conc: pd.DataFrame, markers: pd.DataFrame) -> None:
    lines = []
    lines.append("# Report-ready biomarker numbers\n\n")
    lines.append("## Label audit\n\n")
    lines.append("Paper table counts are LumA 248, LumB 98, Basal 112, HER2 53.\n\n")
    lines.append("The current DE file already uses those names. Step-3 Enrichr tags do not: ")
    lines.append("its HER2 list is the n=112 (Basal) gene set and its Basal list is the n=53 (HER2) gene set. ")
    lines.append("Enrichment tables below are retagged to the paper names.\n\n")
    lines.append(md_table(audit))
    lines.append("\n\n")
    erbb2 = markers[(markers["omic"] == "mRNA") & (markers["gene"].isin(["ESR1", "ERBB2", "GRB7", "KRT5", "FOXC1", "MKI67"]))]
    if not erbb2.empty:
        lines.append("Mean centered mRNA of canonical markers by paper name:\n\n")
        wide = erbb2.pivot(index="gene", columns="paper_name", values="mean").reindex(columns=PAPER_ORDER)
        lines.append(md_table(wide.reset_index()))
        lines.append("\n\n")
    lines.append("## Cohort\n\n")
    lines.append(f"- Consensus features: **{len(de)}**\n")
    lines.append(f"- DE pass (`q<0.05` and `|d|≥0.3`): **{int(de['de_pass'].sum())}** / {len(de)}\n\n")
    summary = (
        de.groupby("class_name")
        .agg(n_consensus=("feature", "size"), n_de_pass=("de_pass", "sum"))
        .reindex(PAPER_ORDER)
        .reset_index()
    )
    lines.append(md_table(summary))
    lines.append("\n\n## Effect sizes called out in the draft\n\n")
    focus = de[de["feature"].isin(["GAS2L3", "TBC1D31", "CEP55", "SGO1", "INPP4B", "P53", "EEF2"])]
    cols = ["class_name", "feature", "omic", "n_folds", "cohens_d", "q_value", "de_pass"]
    lines.append(md_table(focus[cols].sort_values(["feature", "class_name"])))
    lines.append("\n\n## CNV ∩ mRNA (same symbol, same subtype, both in consensus)\n\n")
    if conc.empty:
        lines.append("_No shared symbols._\n")
    else:
        lines.append(md_table(conc))
        lines.append("\n")
    gate_path = OUT / "gate_summary.csv"
    if gate_path.is_file():
        lines.append("\n## BRCA gated fusion (validation folds)\n\n")
        lines.append(
            "Gate is the weight on the OCA-enhanced latent: "
            "`fused = gate * enhanced + (1 - gate) * original`. "
            "Values are the mean across latent dimensions, then the mean ± SD across 5 folds. "
            "This run is BRCA only.\n\n"
        )
        lines.append(md_table(pd.read_csv(gate_path)))
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    X, y, feature_names = load_full_cohort(cfg, fold_one_based=1)
    de_raw = pd.read_csv(DE_CSV)
    audit = audit_labels(y, de_raw)
    audit.to_csv(OUT / "label_audit.csv", index=False)
    print(audit.to_string(index=False))

    markers = marker_means(y, feature_names, X, audit)
    markers.to_csv(OUT / "marker_means.csv", index=False)
    plot_marker_heatmap(markers, OUT / "marker_means_heatmap.png")

    de = de_raw.copy()
    de["de_pass"] = de["de_pass"].astype(str).str.lower().isin(["true", "1"])
    de = apply_paper_names(de, audit)
    de.to_csv(OUT / "consensus_with_de_paper_names.csv", index=False)
    paper_cols = [
        "class_name",
        "de_csv_name",
        "omic",
        "feature",
        "n_folds",
        "mean_abs_grad",
        "cohens_d",
        "q_value",
        "de_pass",
        "mean_diff",
    ]
    de[paper_cols].sort_values(["class_name", "omic", "cohens_d"]).to_csv(
        OUT / "table_consensus_de.csv", index=False
    )
    plot_cohens_d(de, OUT / "cohens_d_by_subtype.png")

    enr = pd.read_csv(ENR_CSV)
    enr = retag_enrichment(enr)
    enr.to_csv(OUT / "enrichment_paper_names.csv", index=False)
    plot_enrichment(enr, OUT / "enrichment_luma.png")
    luma_sig = enr[(enr["tag"] == "LumA_consensus") & (enr["Adjusted P-value"] < 0.05)].copy()
    luma_sig = luma_sig.sort_values(["Gene_set", "Adjusted P-value"])
    keep = [
        "tag",
        "Gene_set",
        "Term",
        "Overlap",
        "P-value",
        "Adjusted P-value",
        "Genes",
    ]
    luma_sig[keep].to_csv(OUT / "table_luma_enrichment_fdr05.csv", index=False)

    conc = concordance_table(de)
    conc.to_csv(OUT / "cnv_mrna_concordance.csv", index=False)
    plot_concordance(conc, OUT / "cnv_mrna_concordance_her2.png")

    write_numbers(OUT / "NUMBERS.md", audit, de, conc, markers)
    print(f"Wrote assets to {OUT}")


if __name__ == "__main__":
    main()
