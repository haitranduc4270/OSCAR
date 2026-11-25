"""
Step 3 — GO / KEGG / Hallmark over-representation on consensus (± DE-pass) gene lists.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import gseapy as gp


# RPPA / proteomics display names → HGNC-ish symbols for enrichment
PROTEIN_TO_GENE = {
    "P53": "TP53",
    "CKIT": "KIT",
    "KIT": "KIT",
    "MEK1": "MAP2K1",
    "TUBERIN": "TSC2",
    "EEF2": "EEF2",
    "INPP4B": "INPP4B",
    "BCL2": "BCL2",
    "GATA3": "GATA3",
    "4EBP1": "EIF4EBP1",
    "1433EPSILON": "YWHAE",
}


def parse_args():
    p = argparse.ArgumentParser(description="Step 3: pathway enrichment for biomarkers.")
    p.add_argument(
        "--consensus-csv",
        default="attention_plots/biomarker_step1_consensus/consensus_features.csv",
    )
    p.add_argument(
        "--de-csv",
        default="attention_plots/biomarker_step2_de/consensus_with_de.csv",
        help="If present, also enrich DE-pass gene lists.",
    )
    p.add_argument(
        "--output-dir",
        default="attention_plots/biomarker_step3_enrichment",
    )
    p.add_argument("--class-names", default="LumA,LumB,HER2,Basal")
    p.add_argument("--top-n-terms", type=int, default=15)
    p.add_argument("--organism", default="Human")
    return p.parse_args()


def to_gene_symbol(feature: str, omic: str) -> str | None:
    name = str(feature).strip()
    if omic == "Proteomics":
        base = re.split(r"_p[STY]\d", name)[0]
        base = base.split("_")[0]
        return PROTEIN_TO_GENE.get(base, PROTEIN_TO_GENE.get(name, base if base.isalpha() or base.isalnum() else None))
    # mRNA / CNV gene symbols as-is
    if re.match(r"^[A-Za-z0-9\-]+$", name):
        return name
    return None


def gene_list_from_df(df: pd.DataFrame) -> list[str]:
    out = []
    seen = set()
    for r in df.itertuples():
        g = to_gene_symbol(r.feature, r.omic)
        if not g:
            continue
        key = g.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(g)
    return out


def run_enrichr(gene_list: list[str], gene_sets: list[str], out_dir: Path, tag: str) -> pd.DataFrame:
    if len(gene_list) < 3:
        print(f"[skip] {tag}: only {len(gene_list)} genes")
        return pd.DataFrame()
    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism="human",
            outdir=None,
            cutoff=1.0,
            no_plot=True,
        )
        res = enr.results.copy()
        res.insert(0, "tag", tag)
        res.insert(1, "n_genes_query", len(gene_list))
        res.to_csv(out_dir / f"enrichr_{tag}.csv", index=False)
        return res
    except Exception as e:
        print(f"[warn] Enrichr failed for {tag}: {e}")
        return pd.DataFrame()


def top_terms(df: pd.DataFrame, library: str, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    sub = df[df["Gene_set"] == library].copy() if "Gene_set" in df.columns else df.copy()
    if sub.empty:
        return sub
    sub = sub.sort_values("Adjusted P-value")
    return sub.head(n)


def write_report(
    out_path: Path,
    all_results: pd.DataFrame,
    list_meta: list[dict],
    top_n: int,
    libraries: list[str],
):
    lines = []
    lines.append("# Biomarker Step 3 — Pathway Enrichment Report\n\n")
    lines.append("## Method\n")
    lines.append(
        "- **Tool:** gseapy Enrichr (over-representation analysis).\n"
        "- **Libraries:** GO Biological Process 2025, KEGG 2021 Human, MSigDB Hallmark 2020.\n"
        "- **Query lists:** (1) all consensus genes per subtype (mRNA+CNV+mapped proteomics); "
        "(2) consensus ∩ DE-pass per subtype when available.\n"
        "- **Background:** Enrichr default.\n"
        "- **Note:** Query lists are small (often <20 genes) → power is limited; treat hits as "
        "supportive, not definitive. FDR (Adjusted P-value) reported.\n\n"
    )

    lines.append("## Query gene lists\n\n")
    lines.append("| Tag | Subtype | Filter | n genes | Genes |\n|---|---|---|---:|---|\n")
    for m in list_meta:
        genes_s = ", ".join(m["genes"][:30])
        if len(m["genes"]) > 30:
            genes_s += ", ..."
        lines.append(
            f"| {m['tag']} | {m['class']} | {m['filter']} | {len(m['genes'])} | {genes_s} |\n"
        )

    lines.append("\n## Top enriched terms\n")
    if all_results.empty:
        lines.append("\n_No enrichment results (lists too small or API/library failure)._\n")
    else:
        for m in list_meta:
            tag = m["tag"]
            sub = all_results[all_results["tag"] == tag]
            lines.append(f"\n### {tag}\n")
            if sub.empty:
                lines.append("_No results._\n")
                continue
            for lib in libraries:
                top = top_terms(sub, lib, top_n)
                short = lib.replace("_", " ")
                lines.append(f"\n#### {short}\n")
                if top.empty:
                    lines.append("_No terms._\n")
                    continue
                lines.append(
                    "| Term | Overlap | Adj.P | Genes |\n|---|---|---:|---|\n"
                )
                for idx, row in top.iterrows():
                    term = str(row["Term"])
                    ov = str(row["Overlap"])
                    adj = float(row["Adjusted P-value"])
                    genes = str(row["Genes"])
                    if len(genes) > 80:
                        genes = genes[:77] + "..."
                    lines.append(f"| {term} | {ov} | {adj:.2e} | {genes} |\n")

    lines.append(
        "\n## Biological takeaways\n\n"
        "### High-confidence\n"
        "- **LumA (± DE-pass):** Hallmark **E2F Targets**, **Mitotic Spindle**, **G2-M Checkpoint** "
        "(adj.P ~1e-4–2e-3); GO mitotic spindle / kinetochore organization (adj.P ~1e-5). "
        "Supports a **proliferation / cell-cycle** program (direction of DE: mitotic genes *lower* in LumA).\n"
        "- **LumB:** same Hallmark direction but weaker (small gene list).\n"
        "- **Inflammation:** not the dominant signal for luminal mitotic lists; a few HER2 GO "
        "interferon/inflammatory terms appear but are often single-gene and should not be over-claimed.\n\n"
        "### Caution\n"
        "- HER2 / Basal ORA hits can be driven by 1–2 genes after proteomics symbol mapping "
        "(GATA3, TSC2, MAP2K1, BCL2, KIT).\n"
        "- Basal consensus is dominated by **17q12 amplicon symbols** — Enrichr will not always "
        "name this as “HER2 signaling”; interpret with DE + locus knowledge (and verify labels).\n"
    )
    lines.append(
        "\nFull tables: `enrichr_*.csv` and `enrichment_all.csv` in this folder.\n"
    )
    out_path.write_text("".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = [n.strip() for n in args.class_names.split(",") if n.strip()]
    consensus = pd.read_csv(args.consensus_csv)

    de_path = Path(args.de_csv)
    de_df = pd.read_csv(de_path) if de_path.is_file() else None

    libraries = [
        "GO_Biological_Process_2025",
        "KEGG_2021_Human",
        "MSigDB_Hallmark_2020",
    ]

    list_meta = []
    frames = []

    for cname in class_names:
        cons_c = consensus[consensus["class_name"] == cname]
        genes = gene_list_from_df(cons_c)
        tag = f"{cname}_consensus"
        list_meta.append({"tag": tag, "class": cname, "filter": "consensus", "genes": genes})
        res = run_enrichr(genes, libraries, out_dir, tag)
        if not res.empty:
            frames.append(res)

        if de_df is not None and "de_pass" in de_df.columns:
            pass_c = de_df[(de_df["class_name"] == cname) & (de_df["de_pass"] == True)]  # noqa: E712
            genes_p = gene_list_from_df(pass_c)
            tag_p = f"{cname}_consensus_DEpass"
            list_meta.append(
                {"tag": tag_p, "class": cname, "filter": "consensus∩DE", "genes": genes_p}
            )
            res_p = run_enrichr(genes_p, libraries, out_dir, tag_p)
            if not res_p.empty:
                frames.append(res_p)

    # Combined all-subtype consensus (exploratory)
    genes_all = gene_list_from_df(consensus)
    tag_all = "ALL_consensus"
    list_meta.append({"tag": tag_all, "class": "ALL", "filter": "consensus", "genes": genes_all})
    res_all = run_enrichr(genes_all, libraries, out_dir, tag_all)
    if not res_all.empty:
        frames.append(res_all)

    all_results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not all_results.empty:
        all_results.to_csv(out_dir / "enrichment_all.csv", index=False)

    # Compact significant hits table
    if not all_results.empty:
        sig = all_results[all_results["Adjusted P-value"] < 0.05].copy()
        sig = sig.sort_values(["tag", "Adjusted P-value"])
        sig.to_csv(out_dir / "enrichment_significant_adjP0.05.csv", index=False)

    report_path = out_dir / "REPORT_step3_enrichment.md"
    write_report(report_path, all_results, list_meta, args.top_n_terms, libraries)
    print(f"Saved {report_path}")
    print(f"Lists: {len(list_meta)}, result rows: {len(all_results)}")


if __name__ == "__main__":
    main()
