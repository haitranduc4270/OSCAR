"""Frozen Hallmark and RPPA program definitions for the pathway-token model.

Hallmark v7.4 is the release whose symbol union is 4,383 genes. A gene enters
the shared embedding table only when that symbol is measured on both CNV and
mRNA. ERBB2, GRB7 and STARD3 stay inside any Hallmark set they already belong
to, and also form the extra pathway ERBB2_AMPLICON.

Phospho antibodies are those whose RPPA name contains ``_p``. Cleaved
caspases do not, so inside the apoptosis program they join the total mean.
"""

from __future__ import annotations

from pathlib import Path

ERBB2_PATHWAY = "ERBB2_AMPLICON"
ERBB2_GENES = ("ERBB2", "GRB7", "STARD3")

# Program name, member antibodies, Hallmark pathways the program may read.
# ``None`` members means "every antibody not named above".
PROTEIN_PROGRAMS: list[tuple[str, tuple[str, ...] | None, tuple[str, ...]]] = [
    (
        "hormone",
        (
            "ERALPHA",
            "ERALPHA_pS118",
            "PR",
            "AR",
            "GATA3",
            "BCL2",
            "INPP4B",
        ),
        (
            "HALLMARK_ESTROGEN_RESPONSE_EARLY",
            "HALLMARK_ESTROGEN_RESPONSE_LATE",
        ),
    ),
    (
        "pi3k_akt_mtor",
        (
            "AKT",
            "AKT_pS473",
            "AKT_pT308",
            "PI3KP110ALPHA",
            "PI3KP85",
            "PTEN",
            "PDK1",
            "PDK1_pS241",
            "MTOR",
            "MTOR_pS2448",
            "RICTOR",
            "RICTOR_pT1135",
            "RAPTOR",
            "TSC1",
            "TUBERIN",
            "TUBERIN_pT1462",
            "PRAS40_pT246",
            "P70S6K1",
            "P70S6K_pT389",
            "S6",
            "S6_pS235S236",
            "S6_pS240S244",
            "4EBP1",
            "4EBP1_pS65",
            "4EBP1_pT37T46",
            "4EBP1_pT70",
            "EIF4E",
            "EIF4G",
            "GSK3ALPHABETA",
            "GSK3ALPHABETA_pS21S9",
            "GSK3_pS9",
            "FOXO3A",
            "FOXO3A_pS318S321",
            "IRS1",
            "NDRG1_pT346",
            "GAB2",
            "PKCALPHA",
            "PKCALPHA_pS657",
            "PKCPANBETAII_pS660",
        ),
        ("HALLMARK_PI3K_AKT_MTOR_SIGNALING",),
    ),
    (
        "rtk",
        (
            "EGFR",
            "EGFR_pY1068",
            "EGFR_pY1173",
            "HER2",
            "HER2_pY1248",
            "HER3",
            "HER3_pY1289",
            "CMET",
            "CMET_pY1235",
            "IGF1R_pY1135Y1136",
            "CKIT",
            "VEGFR2",
            "AXL",
            "SRC",
            "SRC_pY416",
            "SRC_pY527",
            "SHC_pY317",
            "HEREGULIN",
            "MIG6",
        ),
        ("HALLMARK_PI3K_AKT_MTOR_SIGNALING", ERBB2_PATHWAY),
    ),
    (
        "ras_mapk",
        (
            "ARAF",
            "ARAF_pS299",
            "BRAF",
            "BRAF_pS445",
            "NRAS",
            "CRAF",
            "CRAF_pS338",
            "MEK1",
            "MEK1_pS217S221",
            "ERK2",
            "MAPK_pT202Y204",
            "P90RSK",
            "P90RSK_pT359S363",
            "P38MAPK",
            "P38_pT180Y182",
            "JNK2",
            "JNK_pT183Y185",
            "CJUN_pS73",
            "DUSP4",
            "PEA15",
            "PEA15_pS116",
            "ETS1",
            "SHP2_pY542",
        ),
        ("HALLMARK_KRAS_SIGNALING_UP",),
    ),
    (
        "cell_cycle",
        (
            "CDK1",
            "CDK1_pY15",
            "CYCLINB1",
            "CYCLIND1",
            "CYCLINE1",
            "CYCLINE2",
            "P27",
            "P27_pT157",
            "P27_pT198",
            "P21",
            "PCNA",
            "RB",
            "RB_pS807S811",
            "FOXM1",
            "P16INK4A",
            "STATHMIN",
            "CMYC",
        ),
        ("HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT"),
    ),
    (
        "dna_repair",
        (
            "53BP1",
            "ATM",
            "CHK1",
            "CHK1_pS345",
            "CHK1_pS296",
            "CHK2",
            "CHK2_pT68",
            "KU80",
            "MRE11",
            "P53",
            "RAD50",
            "RAD51",
            "XRCC1",
            "BRCA2",
            "ERCC1",
            "MSH2",
            "MSH6",
            "ERCC5",
        ),
        ("HALLMARK_DNA_REPAIR",),
    ),
    (
        "apoptosis",
        (
            "BAK",
            "BAX",
            "BCLXL",
            "BID",
            "BIM",
            "CIAP",
            "BAD_pS112",
            "CASPASE7CLEAVEDD198",
            "CASPASE8",
            "CASPASE3",
            "PARPCLEAVED",
            "SMAC",
            "BCL2A1",
        ),
        ("HALLMARK_APOPTOSIS",),
    ),
    (
        "emt",
        (
            "ECADHERIN",
            "NCADHERIN",
            "PCADHERIN",
            "FIBRONECTIN",
            "COLLAGENVI",
            "CLAUDIN7",
            "PAI1",
            "SNAIL",
            "TRANSGLUTAMINASE",
            "BETACATENIN",
            "CD49B",
        ),
        ("HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",),
    ),
    ("other", None, ()),
]

EXPECTED_PROGRAM_SIZES = {
    "hormone": 7,
    "pi3k_akt_mtor": 39,
    "rtk": 19,
    "ras_mapk": 23,
    "cell_cycle": 17,
    "dna_repair": 18,
    "apoptosis": 13,
    "emt": 11,
    "other": 76,
}


def is_phospho(antibody: str) -> bool:
    return "_p" in antibody


def load_hallmark_gmt(path: str | Path) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = list(dict.fromkeys(parts[2:]))
            sets[name] = genes
    if len(sets) != 50:
        raise ValueError(f"Expected 50 Hallmark sets, found {len(sets)} in {path}")
    return sets


def load_pathformer_pathways(path: str | Path) -> dict[str, list[str]]:
    """Load Pathformer's curated Reactome/KEGG/PID/BioCarta table."""
    sets: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if not header or header[0] != "name":
            raise ValueError(f"Unexpected Pathformer pathway header in {path}: {header[:3]}")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0]:
                continue
            name = parts[0]
            genes = [gene for gene in parts[1].split(",") if gene]
            sets[name] = list(dict.fromkeys(genes))
    if len(sets) != 1497:
        raise ValueError(f"Expected 1497 Pathformer pathways, found {len(sets)} in {path}")
    return sets


# Same RPPA programs, but reading Pathformer pathways closest to the Hallmark links.
PATHFORMER_PROTEIN_READS: dict[str, tuple[str, ...]] = {
    "hormone": (
        "REACTOME_ESTROGEN_DEPENDENT_GENE_EXPRESSION",
        "REACTOME_ESTROGEN_DEPENDENT_NUCLEAR_EVENTS_DOWNSTREAM_OF_ESR_MEMBRANE_SIGNALING",
    ),
    "pi3k_akt_mtor": (
        "REACTOME_MTOR_SIGNALLING",
        "BIOCARTA_AKT_PATHWAY",
    ),
    "rtk": (
        "REACTOME_SIGNALING_BY_ERBB2_IN_CANCER",
        "REACTOME_MTOR_SIGNALLING",
        ERBB2_PATHWAY,
    ),
    "ras_mapk": (
        "BIOCARTA_RAS_PATHWAY",
        "REACTOME_MAP2K_AND_MAPK_ACTIVATION",
    ),
    "cell_cycle": (
        "REACTOME_G2_M_CHECKPOINTS",
        "REACTOME_TRANSCRIPTION_OF_E2F_TARGETS_UNDER_NEGATIVE_CONTROL_BY_P107_RBL1_AND_P130_RBL2_IN_COMPLEX_WITH_HDAC1",
    ),
    "dna_repair": (
        "REACTOME_DISEASES_OF_DNA_REPAIR",
        "KEGG_BASE_EXCISION_REPAIR",
    ),
    "apoptosis": (
        "KEGG_APOPTOSIS",
        "REACTOME_INTRINSIC_PATHWAY_FOR_APOPTOSIS",
    ),
    "emt": (
        "REACTOME_TGF_BETA_RECEPTOR_SIGNALING_IN_EMT_EPITHELIAL_TO_MESENCHYMAL_TRANSITION",
    ),
    "other": (),
}


def assign_protein_programs(
    antibodies: list[str],
    reads_by_program: dict[str, tuple[str, ...]] | None = None,
) -> list[tuple[str, list[str], tuple[str, ...]]]:
    """Partition ``antibodies`` into the nine programs, residual last."""
    known = set(antibodies)
    assigned: set[str] = set()
    programs: list[tuple[str, list[str], tuple[str, ...]]] = []
    for name, members, default_reads in PROTEIN_PROGRAMS:
        reads = default_reads if reads_by_program is None else reads_by_program[name]
        if members is None:
            residual = [antibody for antibody in antibodies if antibody not in assigned]
            programs.append((name, residual, reads))
            continue
        missing = [antibody for antibody in members if antibody not in known]
        if missing:
            raise ValueError(f"Program {name} names antibodies absent from the panel: {missing}")
        overlap = [antibody for antibody in members if antibody in assigned]
        if overlap:
            raise ValueError(f"Program {name} repeats antibodies: {overlap}")
        assigned.update(members)
        programs.append((name, list(members), reads))

    sizes = {name: len(members) for name, members, _reads in programs}
    if sizes != EXPECTED_PROGRAM_SIZES:
        raise ValueError(f"Program sizes {sizes} do not match {EXPECTED_PROGRAM_SIZES}")
    if sum(sizes.values()) != len(antibodies):
        raise ValueError("Protein programs do not cover the antibody panel")
    return programs
