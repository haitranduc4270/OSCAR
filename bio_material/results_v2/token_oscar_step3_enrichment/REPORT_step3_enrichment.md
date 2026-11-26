# Biomarker Step 3 — Pathway Enrichment Report

## Method
- **Tool:** gseapy Enrichr (over-representation analysis).
- **Libraries:** GO Biological Process 2025, KEGG 2021 Human, MSigDB Hallmark 2020.
- **Query lists:** (1) all consensus genes per subtype (mRNA+CNV+mapped proteomics); (2) consensus ∩ DE-pass per subtype when available.
- **Background:** Enrichr default.
- **Note:** Query lists are small (often <20 genes) → power is limited; treat hits as supportive, not definitive. FDR (Adjusted P-value) reported.

## Query gene lists

| Tag | Subtype | Filter | n genes | Genes |
|---|---|---|---:|---|
| LumA_consensus | LumA | consensus | 22 | DIP2A, PXDNL, HER2, CHK1, P27, AKT, ERALPHA, IGFBP2, G6PD, P90RSK, ACC, S6, TBC1D31, KRT6B, KRT17, KRT5, KRT14, DEPDC1B, GTSE1, NDC80, RNASE7, DSG3 |
| LumA_consensus_DEpass | LumA | consensus∩DE | 12 | TBC1D31, KRT14, DEPDC1B, GTSE1, NDC80, RNASE7, CHK1, P27, ERALPHA, IGFBP2, G6PD, S6 |
| LumB_consensus | LumB | consensus | 17 | TCAP, STARD3, ERBB2, GRB7, IKZF3, PGAP3, PPP1R1B, ERALPHA, P27, ERK2, HER2, EGFR, SHC, GATA3, KRT17, KRT5, TBC1D31 |
| LumB_consensus_DEpass | LumB | consensus∩DE | 6 | KRT17, KRT5, TBC1D31, ERALPHA, SHC, GATA3 |
| HER2_consensus | HER2 | consensus | 22 | GRB7, TCAP, STARD3, IKZF3, CDK12, GATA3, AR, HER2, G6PD, ERALPHA, AMPKALPHA, NOTCH1, ARAF, IRS1, EGFR, KIT, CHK2, RB, ERBB2, TPBG, CCDC6, ESR1 |
| HER2_consensus_DEpass | HER2 | consensus∩DE | 18 | ERBB2, GRB7, TPBG, CCDC6, TCAP, ESR1, STARD3, IKZF3, CDK12, GATA3, HER2, G6PD, ERALPHA, ARAF, IRS1, EGFR, KIT, RB |
| Basal_consensus | Basal | consensus | 32 | ERBB2, GRB7, TCAP, STARD3, PPP1R1B, PGAP3, CDK12, ZPBP2, IKZF3, CSF3, GSDMB, CACNA1G, PNMT, HER2, G6PD, EGFR, P27, KIT, S6, HER3, ACC, ERALPHA, ARAF, STAT3, ERK2, VEGFR2, CD49B, FOXC1, ROS1, KRT6B, ... |
| Basal_consensus_DEpass | Basal | consensus∩DE | 17 | STARD3, FOXC1, ROS1, KRT6B, PNMT, ERBB2, PGAP3, MIEN1, TCAP, HER2, P27, KIT, ACC, ERALPHA, ARAF, ERK2, VEGFR2 |
| ALL_consensus | ALL | consensus | 58 | DIP2A, PXDNL, HER2, CHK1, P27, AKT, ERALPHA, IGFBP2, G6PD, P90RSK, ACC, S6, TBC1D31, KRT6B, KRT17, KRT5, KRT14, DEPDC1B, GTSE1, NDC80, RNASE7, DSG3, TCAP, STARD3, ERBB2, GRB7, IKZF3, PGAP3, PPP1R1B, ERK2, ... |

## Top enriched terms

### LumA_consensus

#### GO Biological Process 2025
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Intermediate Filament Organization (GO:0045109) | 4/72 | 1.12e-04 | KRT17;KRT14;KRT5;KRT6B |
| Supramolecular Fiber Organization (GO:0097435) | 5/339 | 1.47e-03 | KRT17;KRT14;KRT5;DSG3;KRT6B |
| Chromosome Organization (GO:0051276) | 2/47 | 4.20e-02 | RNASE7;PXDNL |
| Hydrogen Peroxide Catabolic Process (GO:0042744) | 1/20 | 7.55e-02 | PXDNL |
| Regulation of Spindle Organization (GO:0090224) | 1/20 | 7.55e-02 | RNASE7 |
| Regulation of Mitotic Cell Cycle Spindle Assembly Checkpoint (GO:0090266) | 1/20 | 7.55e-02 | NDC80 |
| Regulation of Insulin-Like Growth Factor Receptor Signaling Pathway (GO:0043567) | 1/19 | 7.55e-02 | IGFBP2 |
| Nuclear Chromosome Segregation (GO:0098813) | 1/18 | 7.55e-02 | NDC80 |
| Regulation of Bicellular Tight Junction Assembly (GO:2000810) | 1/18 | 7.55e-02 | DSG3 |
| Establishment of Chromosome Localization (GO:0051303) | 1/18 | 7.55e-02 | NDC80 |
| acetyl-CoA Metabolic Process (GO:0006084) | 1/17 | 7.55e-02 | DIP2A |
| Attachment of Mitotic Spindle Microtubules to Kinetochore (GO:0051315) | 1/16 | 7.55e-02 | NDC80 |
| Dendritic Spine Morphogenesis (GO:0060997) | 1/15 | 7.55e-02 | DIP2A |
| acyl-CoA Biosynthetic Process (GO:0071616) | 1/14 | 7.55e-02 | DIP2A |
| acetyl-CoA Biosynthetic Process (GO:0006085) | 1/12 | 7.55e-02 | DIP2A |

#### KEGG 2021 Human
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Staphylococcus aureus infection | 2/95 | 3.39e-02 | KRT17;KRT14 |
| Estrogen signaling pathway | 2/137 | 3.44e-02 | KRT17;KRT14 |
| Pentose phosphate pathway | 1/30 | 7.58e-02 | G6PD |
| Glutathione metabolism | 1/57 | 9.02e-02 | G6PD |
| Central carbon metabolism in cancer | 1/70 | 9.02e-02 | G6PD |
| p53 signaling pathway | 1/73 | 9.02e-02 | GTSE1 |
| Diabetic cardiomyopathy | 1/203 | 2.01e-01 | G6PD |

#### MSigDB Hallmark 2020
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| KRAS Signaling Dn | 2/200 | 1.81e-01 | IGFBP2;KRT5 |
| Reactive Oxygen Species Pathway | 1/49 | 1.98e-01 | G6PD |
| UV Response Up | 1/158 | 1.98e-01 | IGFBP2 |
| Mitotic Spindle | 1/199 | 1.98e-01 | NDC80 |
| Glycolysis | 1/200 | 1.98e-01 | G6PD |
| p53 Pathway | 1/200 | 1.98e-01 | KRT17 |
| G2-M Checkpoint | 1/200 | 1.98e-01 | NDC80 |
| mTORC1 Signaling | 1/200 | 1.98e-01 | G6PD |
| Epithelial Mesenchymal Transition | 1/200 | 1.98e-01 | IGFBP2 |

### LumA_consensus_DEpass

#### GO Biological Process 2025
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Transcription-Dependent Tethering of RNA Polymerase II Gene DNA at Nuclear Periphery (GO:0000972) | 1/6 | 6.47e-02 | RNASE7 |
| Nuclear Transport (GO:0051169) | 1/38 | 6.47e-02 | RNASE7 |
| Establishment of Spindle Orientation (GO:0051294) | 1/40 | 6.47e-02 | NDC80 |
| Regulation of Mitotic Spindle Organization (GO:0060236) | 1/41 | 6.47e-02 | RNASE7 |
| Defense Response to Fungus (GO:0050832) | 1/42 | 6.47e-02 | RNASE7 |
| RNA Transport (GO:0050658) | 1/43 | 6.47e-02 | RNASE7 |
| Chromosome Organization (GO:0051276) | 1/47 | 6.47e-02 | RNASE7 |
| Establishment of Mitotic Spindle Localization (GO:0040001) | 1/36 | 6.47e-02 | NDC80 |
| Mitotic Nuclear Division (GO:0140014) | 1/47 | 6.47e-02 | NDC80 |
| Nucleocytoplasmic Transport (GO:0006913) | 1/50 | 6.47e-02 | RNASE7 |
| Signal Transduction by P53 Class Mediator (GO:0072331) | 1/52 | 6.47e-02 | GTSE1 |
| Antibacterial Humoral Response (GO:0019731) | 1/53 | 6.47e-02 | RNASE7 |
| Metaphase Chromosome Alignment (GO:0051310) | 1/55 | 6.47e-02 | NDC80 |
| Microtubule Cytoskeleton Organization Involved in Mitosis (GO:1902850) | 1/59 | 6.47e-02 | NDC80 |
| mRNA Transport (GO:0051028) | 1/59 | 6.47e-02 | RNASE7 |

#### KEGG 2021 Human
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Pentose phosphate pathway | 1/30 | 7.51e-02 | G6PD |
| Glutathione metabolism | 1/57 | 7.51e-02 | G6PD |
| Central carbon metabolism in cancer | 1/70 | 7.51e-02 | G6PD |
| p53 signaling pathway | 1/73 | 7.51e-02 | GTSE1 |
| Staphylococcus aureus infection | 1/95 | 7.78e-02 | KRT14 |
| Estrogen signaling pathway | 1/137 | 9.24e-02 | KRT14 |
| Diabetic cardiomyopathy | 1/203 | 1.15e-01 | G6PD |

#### MSigDB Hallmark 2020
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Reactive Oxygen Species Pathway | 1/49 | 1.14e-01 | G6PD |
| UV Response Up | 1/158 | 1.14e-01 | IGFBP2 |
| Mitotic Spindle | 1/199 | 1.14e-01 | NDC80 |
| KRAS Signaling Dn | 1/200 | 1.14e-01 | IGFBP2 |
| Glycolysis | 1/200 | 1.14e-01 | G6PD |
| G2-M Checkpoint | 1/200 | 1.14e-01 | NDC80 |
| mTORC1 Signaling | 1/200 | 1.14e-01 | G6PD |
| Epithelial Mesenchymal Transition | 1/200 | 1.14e-01 | IGFBP2 |

### LumB_consensus

#### GO Biological Process 2025
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Phosphatidylinositol 3-Kinase/Protein Kinase B Signal Transduction (GO:0043491) | 3/39 | 8.90e-04 | ERBB2;GATA3;EGFR |
| ERBB2-EGFR Signaling Pathway (GO:0038134) | 2/5 | 8.90e-04 | ERBB2;EGFR |
| Epidermal Growth Factor Receptor Signaling Pathway (GO:0007173) | 3/49 | 8.90e-04 | GRB7;ERBB2;EGFR |
| ERBB Signaling Pathway (GO:0038127) | 3/58 | 1.11e-03 | GRB7;ERBB2;EGFR |
| ERBB2 Signaling Pathway (GO:0038128) | 2/13 | 3.07e-03 | ERBB2;EGFR |
| Cellular Response to Epidermal Growth Factor Stimulus (GO:0071364) | 2/28 | 1.23e-02 | ERBB2;EGFR |
| Response to Epidermal Growth Factor (GO:0070849) | 2/31 | 1.30e-02 | ERBB2;EGFR |
| Peptidyl-Tyrosine Modification (GO:0018212) | 2/46 | 2.32e-02 | ERBB2;EGFR |
| Positive Regulation of miRNA Transcription (GO:1902895) | 2/47 | 2.32e-02 | GATA3;EGFR |
| Positive Regulation of miRNA Metabolic Process (GO:2000630) | 2/53 | 2.66e-02 | GATA3;EGFR |
| Peptidyl-Tyrosine Phosphorylation (GO:0018108) | 2/58 | 2.89e-02 | ERBB2;EGFR |
| Regulation of miRNA Transcription (GO:1902893) | 2/66 | 3.43e-02 | GATA3;EGFR |
| Intermediate Filament Organization (GO:0045109) | 2/72 | 3.62e-02 | KRT17;KRT5 |
| Positive Regulation of Protein Serine/Threonine Kinase Activity (GO:0071902) | 2/75 | 3.62e-02 | ERBB2;EGFR |
| Positive Regulation of Protein Localization to Membrane (GO:1905477) | 2/76 | 3.62e-02 | ERBB2;EGFR |

#### KEGG 2021 Human
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Bladder cancer | 2/41 | 1.74e-02 | ERBB2;EGFR |
| Endometrial cancer | 2/58 | 1.74e-02 | ERBB2;EGFR |
| Central carbon metabolism in cancer | 2/70 | 1.74e-02 | ERBB2;EGFR |
| Adherens junction | 2/71 | 1.74e-02 | ERBB2;EGFR |
| Non-small cell lung cancer | 2/72 | 1.74e-02 | ERBB2;EGFR |
| Pancreatic cancer | 2/76 | 1.74e-02 | ERBB2;EGFR |
| ErbB signaling pathway | 2/85 | 1.86e-02 | ERBB2;EGFR |
| Prostate cancer | 2/97 | 2.11e-02 | ERBB2;EGFR |
| Parathyroid hormone synthesis, secretion and action | 2/106 | 2.12e-02 | GATA3;EGFR |
| HIF-1 signaling pathway | 2/109 | 2.12e-02 | ERBB2;EGFR |
| Breast cancer | 2/147 | 3.00e-02 | ERBB2;EGFR |
| Gastric cancer | 2/149 | 3.00e-02 | ERBB2;EGFR |
| Estrogen signaling pathway | 2/137 | 3.00e-02 | KRT17;EGFR |
| Focal adhesion | 2/201 | 4.80e-02 | ERBB2;EGFR |
| Proteoglycans in cancer | 2/205 | 4.80e-02 | ERBB2;EGFR |

#### MSigDB Hallmark 2020
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Apical Junction | 2/200 | 1.57e-01 | GRB7;EGFR |
| Apical Surface | 1/44 | 1.57e-01 | GATA3 |
| Protein Secretion | 1/96 | 1.57e-01 | EGFR |
| PI3K/AKT/mTOR  Signaling | 1/105 | 1.57e-01 | EGFR |
| UV Response Dn | 1/144 | 1.57e-01 | ERBB2 |
| UV Response Up | 1/158 | 1.57e-01 | STARD3 |
| Apoptosis | 1/161 | 1.57e-01 | ERBB2 |
| Hypoxia | 1/200 | 1.57e-01 | EGFR |
| Glycolysis | 1/200 | 1.57e-01 | EGFR |
| p53 Pathway | 1/200 | 1.57e-01 | KRT17 |
| Allograft Rejection | 1/200 | 1.57e-01 | EGFR |
| Myogenesis | 1/200 | 1.57e-01 | TCAP |
| KRAS Signaling Dn | 1/200 | 1.57e-01 | KRT5 |
| Complement | 1/200 | 1.57e-01 | GATA3 |

### LumB_consensus_DEpass

#### GO Biological Process 2025
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Intermediate Filament Organization (GO:0045109) | 2/72 | 2.18e-02 | KRT17;KRT5 |
| Cellular Response to Interleukin-4 (GO:0071353) | 1/13 | 2.18e-02 | GATA3 |
| Negative Regulation of Fibroblast Growth Factor Receptor Signaling Pathway (GO:0040037) | 1/13 | 2.18e-02 | GATA3 |
| Cardiac Right Ventricle Morphogenesis (GO:0003215) | 1/13 | 2.18e-02 | GATA3 |
| Regulation of Interleukin-5 Production (GO:0032674) | 1/12 | 2.18e-02 | GATA3 |
| Negative Regulation of Locomotion (GO:0040013) | 1/12 | 2.18e-02 | GATA3 |
| Catecholamine Biosynthetic Process (GO:0042423) | 1/11 | 2.18e-02 | GATA3 |
| Mesonephros Development (GO:0001823) | 1/10 | 2.18e-02 | GATA3 |
| Cellular Response to Interferon-Alpha (GO:0035457) | 1/9 | 2.18e-02 | GATA3 |
| Response to Interleukin-4 (GO:0070670) | 1/9 | 2.18e-02 | GATA3 |
| Positive Regulation of Epithelial Tube Formation (GO:1905278) | 1/8 | 2.18e-02 | GATA3 |
| Mesenchymal to Epithelial Transition (GO:0060231) | 1/10 | 2.18e-02 | GATA3 |
| Regulation of Mammary Gland Epithelial Cell Proliferation (GO:0033599) | 1/7 | 2.18e-02 | GATA3 |
| Regulation of Cell Motility (GO:2000145) | 2/126 | 2.18e-02 | KRT5;GATA3 |
| Regulation of Nephron Tubule Epithelial Cell Differentiation (GO:0072182) | 1/8 | 2.18e-02 | GATA3 |

#### KEGG 2021 Human
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Inflammatory bowel disease | 1/65 | 3.80e-02 | GATA3 |
| Th1 and Th2 cell differentiation | 1/92 | 3.80e-02 | GATA3 |
| Staphylococcus aureus infection | 1/95 | 3.80e-02 | KRT17 |
| Parathyroid hormone synthesis, secretion and action | 1/106 | 3.80e-02 | GATA3 |
| Th17 cell differentiation | 1/107 | 3.80e-02 | GATA3 |
| Estrogen signaling pathway | 1/137 | 4.04e-02 | KRT17 |

#### MSigDB Hallmark 2020
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Apical Surface | 1/44 | 5.25e-02 | GATA3 |
| KRAS Signaling Dn | 1/200 | 5.85e-02 | KRT5 |
| p53 Pathway | 1/200 | 5.85e-02 | KRT17 |
| Complement | 1/200 | 5.85e-02 | GATA3 |

### HER2_consensus

#### GO Biological Process 2025
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Phosphatidylinositol 3-Kinase/Protein Kinase B Signal Transduction (GO:0043491) | 4/39 | 4.52e-05 | IRS1;ERBB2;GATA3;EGFR |
| ERBB2-EGFR Signaling Pathway (GO:0038134) | 2/5 | 2.23e-03 | ERBB2;EGFR |
| Epidermal Growth Factor Receptor Signaling Pathway (GO:0007173) | 3/49 | 2.23e-03 | GRB7;ERBB2;EGFR |
| Positive Regulation of Smooth Muscle Cell Differentiation (GO:0051152) | 2/7 | 2.23e-03 | NOTCH1;KIT |
| + Reg of Phosphatidylinositol 3-Kinase/Prot Kinase B Signal Transduction (GO:0051897) | 4/162 | 2.23e-03 | IRS1;KIT;GATA3;EGFR |
| Positive Regulation of Cell Migration (GO:0030335) | 5/344 | 2.23e-03 | GRB7;NOTCH1;KIT;GATA3;EGFR |
| ERBB Signaling Pathway (GO:0038127) | 3/58 | 2.23e-03 | GRB7;ERBB2;EGFR |
| Intracellular Signaling Cassette (GO:0141124) | 5/354 | 2.23e-03 | IRS1;ERBB2;ARAF;GATA3;EGFR |
| Regulation of miRNA Transcription (GO:1902893) | 3/66 | 2.88e-03 | GATA3;ESR1;EGFR |
| Regulation of Phosphatidylinositol 3-Kinase/Protein Kinase B Signal Transduction (GO:0051896) | 4/219 | 3.82e-03 | IRS1;KIT;GATA3;EGFR |
| ERBB2 Signaling Pathway (GO:0038128) | 2/13 | 3.82e-03 | ERBB2;EGFR |
| Cardiac Cell Development (GO:0055006) | 2/13 | 3.82e-03 | NOTCH1;TCAP |
| Muscle Tissue Morphogenesis (GO:0060415) | 2/17 | 4.99e-03 | NOTCH1;TCAP |
| Positive Regulation of Cell Population Proliferation (GO:0008284) | 5/484 | 4.99e-03 | NOTCH1;IRS1;KIT;ERBB2;EGFR |
| Negative Regulation of Apoptotic Process (GO:0043066) | 5/475 | 4.99e-03 | NOTCH1;ERBB2;ARAF;GATA3;EGFR |

#### KEGG 2021 Human
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Breast cancer | 6/147 | 8.57e-07 | NOTCH1;KIT;ERBB2;ARAF;ESR1;EGFR |
| Pathways in cancer | 8/531 | 2.40e-06 | AR;NOTCH1;KIT;ERBB2;ARAF;CCDC6;ESR1;EGFR |
| Central carbon metabolism in cancer | 4/70 | 2.85e-05 | G6PD;KIT;ERBB2;EGFR |
| Prostate cancer | 4/97 | 7.91e-05 | AR;ERBB2;ARAF;EGFR |
| Bladder cancer | 3/41 | 2.13e-04 | ERBB2;ARAF;EGFR |
| Endometrial cancer | 3/58 | 5.08e-04 | ERBB2;ARAF;EGFR |
| Non-small cell lung cancer | 3/72 | 7.55e-04 | ERBB2;ARAF;EGFR |
| Proteoglycans in cancer | 4/205 | 7.55e-04 | ERBB2;ARAF;ESR1;EGFR |
| Pancreatic cancer | 3/76 | 7.62e-04 | ERBB2;ARAF;EGFR |
| ErbB signaling pathway | 3/85 | 9.58e-04 | ERBB2;ARAF;EGFR |
| Parathyroid hormone synthesis, secretion and action | 3/106 | 1.68e-03 | ARAF;GATA3;EGFR |
| MAPK signaling pathway | 4/294 | 2.01e-03 | KIT;ERBB2;ARAF;EGFR |
| MicroRNAs in cancer | 4/310 | 2.27e-03 | NOTCH1;IRS1;ERBB2;EGFR |
| FoxO signaling pathway | 3/131 | 2.45e-03 | IRS1;ARAF;EGFR |
| PI3K-Akt signaling pathway | 4/354 | 3.13e-03 | IRS1;KIT;ERBB2;EGFR |

#### MSigDB Hallmark 2020
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| UV Response Dn | 3/144 | 1.01e-02 | IRS1;KIT;ERBB2 |
| Glycolysis | 3/200 | 1.01e-02 | G6PD;TPBG;EGFR |
| Apical Junction | 3/200 | 1.01e-02 | GRB7;IRS1;EGFR |
| Hypoxia | 2/200 | 7.73e-02 | TPBG;EGFR |
| Estrogen Response Early | 2/200 | 7.73e-02 | AR;TPBG |
| Myogenesis | 2/200 | 7.73e-02 | NOTCH1;TCAP |
| Notch Signaling | 1/32 | 1.14e-01 | NOTCH1 |
| Wnt-beta Catenin Signaling | 1/42 | 1.21e-01 | NOTCH1 |
| Apical Surface | 1/44 | 1.21e-01 | GATA3 |
| Reactive Oxygen Species Pathway | 1/49 | 1.21e-01 | G6PD |
| Complement | 1/200 | 1.98e-01 | GATA3 |
| Allograft Rejection | 1/200 | 1.98e-01 | EGFR |
| Estrogen Response Late | 1/200 | 1.98e-01 | TPBG |
| Adipogenesis | 1/200 | 1.98e-01 | ARAF |
| p53 Pathway | 1/200 | 1.98e-01 | NOTCH1 |

### HER2_consensus_DEpass

#### GO Biological Process 2025
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Phosphatidylinositol 3-Kinase/Protein Kinase B Signal Transduction (GO:0043491) | 4/39 | 1.46e-05 | IRS1;ERBB2;GATA3;EGFR |
| ERBB2-EGFR Signaling Pathway (GO:0038134) | 2/5 | 9.43e-04 | ERBB2;EGFR |
| Epidermal Growth Factor Receptor Signaling Pathway (GO:0007173) | 3/49 | 9.43e-04 | GRB7;ERBB2;EGFR |
| + Reg of Phosphatidylinositol 3-Kinase/Prot Kinase B Signal Transduction (GO:0051897) | 4/162 | 9.43e-04 | IRS1;KIT;GATA3;EGFR |
| Intracellular Signaling Cassette (GO:0141124) | 5/354 | 9.43e-04 | IRS1;ERBB2;ARAF;GATA3;EGFR |
| ERBB Signaling Pathway (GO:0038127) | 3/58 | 1.20e-03 | GRB7;ERBB2;EGFR |
| Regulation of miRNA Transcription (GO:1902893) | 3/66 | 1.52e-03 | GATA3;ESR1;EGFR |
| Regulation of Phosphatidylinositol 3-Kinase/Protein Kinase B Signal Transduction (GO:0051896) | 4/219 | 1.87e-03 | IRS1;KIT;GATA3;EGFR |
| ERBB2 Signaling Pathway (GO:0038128) | 2/13 | 2.60e-03 | ERBB2;EGFR |
| Cell Surface Receptor Protein Tyrosine Kinase Signaling Pathway (GO:0007169) | 4/296 | 4.81e-03 | GRB7;IRS1;ERBB2;KIT |
| Cellular Response to Estradiol Stimulus (GO:0071392) | 2/21 | 5.70e-03 | ESR1;EGFR |
| Positive Regulation of Cell Migration (GO:0030335) | 4/344 | 6.61e-03 | GRB7;KIT;GATA3;EGFR |
| Positive Regulation of Intracellular Signal Transduction (GO:1902533) | 5/652 | 6.61e-03 | IRS1;ERBB2;KIT;GATA3;EGFR |
| Response to Estrogen (GO:0043627) | 2/32 | 7.01e-03 | GATA3;ESR1 |
| Response to Epidermal Growth Factor (GO:0070849) | 2/31 | 7.01e-03 | ERBB2;EGFR |

#### KEGG 2021 Human
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Breast cancer | 5/147 | 1.37e-05 | ERBB2;KIT;ARAF;ESR1;EGFR |
| Central carbon metabolism in cancer | 4/70 | 1.74e-05 | G6PD;ERBB2;KIT;EGFR |
| Pathways in cancer | 6/531 | 1.37e-04 | ERBB2;KIT;ARAF;CCDC6;ESR1;EGFR |
| Bladder cancer | 3/41 | 1.37e-04 | ERBB2;ARAF;EGFR |
| Endometrial cancer | 3/58 | 3.15e-04 | ERBB2;ARAF;EGFR |
| Proteoglycans in cancer | 4/205 | 4.20e-04 | ERBB2;ARAF;ESR1;EGFR |
| Non-small cell lung cancer | 3/72 | 4.31e-04 | ERBB2;ARAF;EGFR |
| Pancreatic cancer | 3/76 | 4.44e-04 | ERBB2;ARAF;EGFR |
| ErbB signaling pathway | 3/85 | 5.52e-04 | ERBB2;ARAF;EGFR |
| Prostate cancer | 3/97 | 7.36e-04 | ERBB2;ARAF;EGFR |
| Parathyroid hormone synthesis, secretion and action | 3/106 | 8.53e-04 | ARAF;GATA3;EGFR |
| MAPK signaling pathway | 4/294 | 8.53e-04 | ERBB2;KIT;ARAF;EGFR |
| FoxO signaling pathway | 3/131 | 1.38e-03 | IRS1;ARAF;EGFR |
| PI3K-Akt signaling pathway | 4/354 | 1.49e-03 | IRS1;ERBB2;KIT;EGFR |
| Gastric cancer | 3/149 | 1.75e-03 | ERBB2;ARAF;EGFR |

#### MSigDB Hallmark 2020
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| UV Response Dn | 3/144 | 4.56e-03 | IRS1;ERBB2;KIT |
| Glycolysis | 3/200 | 4.56e-03 | G6PD;TPBG;EGFR |
| Apical Junction | 3/200 | 4.56e-03 | GRB7;IRS1;EGFR |
| Hypoxia | 2/200 | 6.51e-02 | TPBG;EGFR |
| Apical Surface | 1/44 | 1.37e-01 | GATA3 |
| Reactive Oxygen Species Pathway | 1/49 | 1.37e-01 | G6PD |
| Complement | 1/200 | 1.66e-01 | GATA3 |
| Myogenesis | 1/200 | 1.66e-01 | TCAP |
| Allograft Rejection | 1/200 | 1.66e-01 | EGFR |
| Estrogen Response Late | 1/200 | 1.66e-01 | TPBG |
| Estrogen Response Early | 1/200 | 1.66e-01 | TPBG |
| Apoptosis | 1/161 | 1.66e-01 | ERBB2 |
| Xenobiotic Metabolism | 1/200 | 1.66e-01 | ESR1 |
| mTORC1 Signaling | 1/200 | 1.66e-01 | G6PD |
| UV Response Up | 1/158 | 1.66e-01 | STARD3 |

### Basal_consensus

#### GO Biological Process 2025
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Cell Surface Receptor Protein Tyrosine Kinase Signaling Pathway (GO:0007169) | 6/296 | 2.34e-03 | GRB7;FOXC1;ERBB2;STAT3;KIT;ROS1 |
| Cellular Response to Epidermal Growth Factor Stimulus (GO:0071364) | 3/28 | 2.34e-03 | FOXC1;ERBB2;EGFR |
| Response to Epidermal Growth Factor (GO:0070849) | 3/31 | 2.34e-03 | FOXC1;ERBB2;EGFR |
| ERBB2-EGFR Signaling Pathway (GO:0038134) | 2/5 | 2.68e-03 | ERBB2;EGFR |
| Phosphatidylinositol 3-Kinase/Protein Kinase B Signal Transduction (GO:0043491) | 3/39 | 2.83e-03 | ERBB2;STAT3;EGFR |
| Positive Regulation of Cell Motility (GO:2000147) | 5/249 | 3.17e-03 | GRB7;MIEN1;STAT3;KIT;EGFR |
| Epidermal Growth Factor Receptor Signaling Pathway (GO:0007173) | 3/49 | 4.03e-03 | GRB7;ERBB2;EGFR |
| ERBB Signaling Pathway (GO:0038127) | 3/58 | 5.85e-03 | GRB7;ERBB2;EGFR |
| ERBB2 Signaling Pathway (GO:0038128) | 2/13 | 8.70e-03 | ERBB2;EGFR |
| Positive Regulation of Cell Migration (GO:0030335) | 5/344 | 8.70e-03 | GRB7;MIEN1;STAT3;KIT;EGFR |
| Protein Phosphorylation (GO:0006468) | 5/373 | 1.15e-02 | ERBB2;KIT;CDK12;ROS1;EGFR |
| Regulation of MAPK Cascade (GO:0043408) | 4/260 | 2.51e-02 | ERBB2;KIT;ROS1;EGFR |
| Cytokine-Mediated Signaling Pathway (GO:0019221) | 4/255 | 2.51e-02 | CSF3;FOXC1;STAT3;KIT |
| Regulation of Cell Migration (GO:0030334) | 5/479 | 2.82e-02 | GRB7;MIEN1;STAT3;KIT;EGFR |
| Positive Regulation of Peptidyl-Serine Phosphorylation (GO:0033138) | 2/30 | 3.02e-02 | ARAF;EGFR |

#### KEGG 2021 Human
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Central carbon metabolism in cancer | 4/70 | 2.02e-04 | G6PD;ERBB2;KIT;EGFR |
| Non-small cell lung cancer | 4/72 | 2.02e-04 | ERBB2;STAT3;ARAF;EGFR |
| Pancreatic cancer | 4/76 | 2.02e-04 | ERBB2;STAT3;ARAF;EGFR |
| Bladder cancer | 3/41 | 9.04e-04 | ERBB2;ARAF;EGFR |
| Breast cancer | 4/147 | 1.47e-03 | ERBB2;KIT;ARAF;EGFR |
| MAPK signaling pathway | 5/294 | 1.47e-03 | ERBB2;KIT;ARAF;EGFR;CACNA1G |
| Endometrial cancer | 3/58 | 1.47e-03 | ERBB2;ARAF;EGFR |
| Acute myeloid leukemia | 3/67 | 1.97e-03 | STAT3;KIT;ARAF |
| Proteoglycans in cancer | 4/205 | 3.19e-03 | ERBB2;STAT3;ARAF;EGFR |
| ErbB signaling pathway | 3/85 | 3.19e-03 | ERBB2;ARAF;EGFR |
| Prostate cancer | 3/97 | 4.28e-03 | ERBB2;ARAF;EGFR |
| HIF-1 signaling pathway | 3/109 | 5.51e-03 | ERBB2;STAT3;EGFR |
| FoxO signaling pathway | 3/131 | 8.66e-03 | STAT3;ARAF;EGFR |
| Pathways in cancer | 5/531 | 9.78e-03 | ERBB2;STAT3;KIT;ARAF;EGFR |
| Gastric cancer | 3/149 | 1.09e-02 | ERBB2;ARAF;EGFR |

#### MSigDB Hallmark 2020
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| UV Response Dn | 2/144 | 1.93e-01 | ERBB2;KIT |
| Glycolysis | 2/200 | 1.93e-01 | G6PD;EGFR |
| Apical Junction | 2/200 | 1.93e-01 | GRB7;EGFR |
| Inflammatory Response | 2/200 | 1.93e-01 | CSF3;ROS1 |
| KRAS Signaling Dn | 1/200 | 2.75e-01 | PNMT |
| Myogenesis | 1/200 | 2.75e-01 | TCAP |
| Allograft Rejection | 1/200 | 2.75e-01 | EGFR |
| Estrogen Response Late | 1/200 | 2.75e-01 | FOXC1 |
| Estrogen Response Early | 1/200 | 2.75e-01 | FOXC1 |
| Adipogenesis | 1/200 | 2.75e-01 | ARAF |
| Apoptosis | 1/161 | 2.75e-01 | ERBB2 |
| Interferon Gamma Response | 1/200 | 2.75e-01 | STAT3 |
| UV Response Up | 1/158 | 2.75e-01 | STARD3 |
| PI3K/AKT/mTOR  Signaling | 1/105 | 2.75e-01 | EGFR |
| Protein Secretion | 1/96 | 2.75e-01 | EGFR |

### Basal_consensus_DEpass

#### GO Biological Process 2025
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Cell Surface Receptor Protein Tyrosine Kinase Signaling Pathway (GO:0007169) | 4/296 | 2.21e-02 | FOXC1;ERBB2;KIT;ROS1 |
| Cellular Response to Epidermal Growth Factor Stimulus (GO:0071364) | 2/28 | 2.39e-02 | FOXC1;ERBB2 |
| Response to Epidermal Growth Factor (GO:0070849) | 2/31 | 2.39e-02 | FOXC1;ERBB2 |
| Cardiac Cell Development (GO:0055006) | 1/13 | 5.55e-02 | TCAP |
| ERBB2 Signaling Pathway (GO:0038128) | 1/13 | 5.55e-02 | ERBB2 |
| Regulation of Hematopoietic Stem Cell Differentiation (GO:1902036) | 1/13 | 5.55e-02 | FOXC1 |
| Muscle Filament Sliding (GO:0030049) | 1/13 | 5.55e-02 | TCAP |
| Embryonic Hemopoiesis (GO:0035162) | 1/13 | 5.55e-02 | KIT |
| Vascular Endothelial Growth Factor Signaling Pathway (GO:0038084) | 1/14 | 5.55e-02 | FOXC1 |
| Response to Chemokine (GO:1990868) | 1/14 | 5.55e-02 | FOXC1 |
| Response to Muscle Stretch (GO:0035994) | 1/14 | 5.55e-02 | TCAP |
| Odontogenesis of Dentin-Containing Tooth (GO:0042475) | 1/15 | 5.55e-02 | FOXC1 |
| Cardiac Myofibril Assembly (GO:0055003) | 1/15 | 5.55e-02 | TCAP |
| Actin-Myosin Filament Sliding (GO:0033275) | 1/15 | 5.55e-02 | TCAP |
| Digestive System Development (GO:0055123) | 1/15 | 5.55e-02 | KIT |

#### KEGG 2021 Human
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Breast cancer | 3/147 | 1.12e-02 | ERBB2;KIT;ARAF |
| Bladder cancer | 2/41 | 1.12e-02 | ERBB2;ARAF |
| Endometrial cancer | 2/58 | 1.12e-02 | ERBB2;ARAF |
| Acute myeloid leukemia | 2/67 | 1.12e-02 | KIT;ARAF |
| Central carbon metabolism in cancer | 2/70 | 1.12e-02 | ERBB2;KIT |
| Non-small cell lung cancer | 2/72 | 1.12e-02 | ERBB2;ARAF |
| MAPK signaling pathway | 3/294 | 1.12e-02 | ERBB2;KIT;ARAF |
| Pancreatic cancer | 2/76 | 1.12e-02 | ERBB2;ARAF |
| ErbB signaling pathway | 2/85 | 1.24e-02 | ERBB2;ARAF |
| Prostate cancer | 2/97 | 1.45e-02 | ERBB2;ARAF |
| Gastric cancer | 2/149 | 3.04e-02 | ERBB2;ARAF |
| Pathways in cancer | 3/531 | 3.84e-02 | ERBB2;KIT;ARAF |
| Proteoglycans in cancer | 2/205 | 4.74e-02 | ERBB2;ARAF |
| Tyrosine metabolism | 1/36 | 1.03e-01 | PNMT |
| PI3K-Akt signaling pathway | 2/354 | 1.14e-01 | ERBB2;KIT |

#### MSigDB Hallmark 2020
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| UV Response Dn | 2/144 | 5.87e-02 | ERBB2;KIT |
| UV Response Up | 1/158 | 1.57e-01 | STARD3 |
| Apoptosis | 1/161 | 1.57e-01 | ERBB2 |
| Myogenesis | 1/200 | 1.57e-01 | TCAP |
| KRAS Signaling Dn | 1/200 | 1.57e-01 | PNMT |
| Adipogenesis | 1/200 | 1.57e-01 | ARAF |
| Estrogen Response Early | 1/200 | 1.57e-01 | FOXC1 |
| Estrogen Response Late | 1/200 | 1.57e-01 | FOXC1 |
| Inflammatory Response | 1/200 | 1.57e-01 | ROS1 |

### ALL_consensus

#### GO Biological Process 2025
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Phosphatidylinositol 3-Kinase/Protein Kinase B Signal Transduction (GO:0043491) | 5/39 | 6.89e-05 | IRS1;ERBB2;STAT3;GATA3;EGFR |
| Cell Surface Receptor Protein Tyrosine Kinase Signaling Pathway (GO:0007169) | 7/296 | 6.88e-03 | GRB7;FOXC1;IRS1;KIT;ERBB2;STAT3;ROS1 |
| Regulation of miRNA Transcription (GO:1902893) | 4/66 | 6.88e-03 | STAT3;GATA3;ESR1;EGFR |
| Intermediate Filament Organization (GO:0045109) | 4/72 | 6.88e-03 | KRT17;KRT14;KRT5;KRT6B |
| Positive Regulation of Cell Migration (GO:0030335) | 7/344 | 6.88e-03 | GRB7;NOTCH1;MIEN1;KIT;STAT3;GATA3;EGFR |
| Regulation of Cell Migration (GO:0030334) | 8/479 | 6.88e-03 | GRB7;NOTCH1;MIEN1;KIT;STAT3;KRT5;DSG3;EGFR |
| Cellular Response to Epidermal Growth Factor Stimulus (GO:0071364) | 3/28 | 6.88e-03 | FOXC1;ERBB2;EGFR |
| ERBB2-EGFR Signaling Pathway (GO:0038134) | 2/5 | 6.88e-03 | ERBB2;EGFR |
| Positive Regulation of Cell Motility (GO:2000147) | 6/249 | 6.88e-03 | GRB7;NOTCH1;MIEN1;KIT;STAT3;EGFR |
| Response to Epidermal Growth Factor (GO:0070849) | 3/31 | 7.24e-03 | FOXC1;ERBB2;EGFR |
| + Reg of Phosphatidylinositol 3-Kinase/Prot Kinase B Signal Transduction (GO:0051897) | 5/162 | 7.24e-03 | CSF3;IRS1;KIT;GATA3;EGFR |
| Response to Estradiol (GO:0032355) | 3/33 | 7.42e-03 | STAT3;ESR1;EGFR |
| Positive Regulation of Smooth Muscle Cell Differentiation (GO:0051152) | 2/7 | 9.92e-03 | NOTCH1;KIT |
| Cellular Response to Cytokine Stimulus (GO:0071345) | 6/289 | 9.97e-03 | CSF3;FOXC1;IRS1;KIT;STAT3;GATA3 |
| Positive Regulation of miRNA Transcription (GO:1902895) | 3/47 | 1.71e-02 | STAT3;GATA3;EGFR |

#### KEGG 2021 Human
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| Breast cancer | 6/147 | 4.75e-04 | NOTCH1;KIT;ERBB2;ARAF;ESR1;EGFR |
| Pathways in cancer | 9/531 | 1.16e-03 | AR;NOTCH1;KIT;ERBB2;STAT3;ARAF;CCDC6;ESR1;EGFR |
| Central carbon metabolism in cancer | 4/70 | 1.58e-03 | G6PD;KIT;ERBB2;EGFR |
| Non-small cell lung cancer | 4/72 | 1.58e-03 | ERBB2;STAT3;ARAF;EGFR |
| Pancreatic cancer | 4/76 | 1.58e-03 | ERBB2;STAT3;ARAF;EGFR |
| Prostate cancer | 4/97 | 3.40e-03 | AR;ERBB2;ARAF;EGFR |
| Bladder cancer | 3/41 | 3.68e-03 | ERBB2;ARAF;EGFR |
| Proteoglycans in cancer | 5/205 | 4.49e-03 | ERBB2;STAT3;ARAF;ESR1;EGFR |
| FoxO signaling pathway | 4/131 | 6.89e-03 | IRS1;STAT3;ARAF;EGFR |
| Endometrial cancer | 3/58 | 6.89e-03 | ERBB2;ARAF;EGFR |
| Estrogen signaling pathway | 4/137 | 6.89e-03 | KRT17;KRT14;ESR1;EGFR |
| Acute myeloid leukemia | 3/67 | 9.15e-03 | KIT;STAT3;ARAF |
| MAPK signaling pathway | 5/294 | 1.40e-02 | KIT;ERBB2;ARAF;EGFR;CACNA1G |
| ErbB signaling pathway | 3/85 | 1.53e-02 | ERBB2;ARAF;EGFR |
| MicroRNAs in cancer | 5/310 | 1.53e-02 | NOTCH1;IRS1;ERBB2;STAT3;EGFR |

#### MSigDB Hallmark 2020
| Term | Overlap | Adj.P | Genes |
|---|---|---:|---|
| UV Response Dn | 3/144 | 9.82e-02 | IRS1;KIT;ERBB2 |
| Estrogen Response Early | 3/200 | 9.82e-02 | AR;FOXC1;TPBG |
| Apical Junction | 3/200 | 9.82e-02 | GRB7;IRS1;EGFR |
| Inflammatory Response | 3/200 | 9.82e-02 | CSF3;TPBG;ROS1 |
| Glycolysis | 3/200 | 9.82e-02 | G6PD;TPBG;EGFR |
| KRAS Signaling Dn | 3/200 | 9.82e-02 | PNMT;IGFBP2;KRT5 |
| UV Response Up | 2/158 | 2.49e-01 | STARD3;IGFBP2 |
| Notch Signaling | 1/32 | 2.49e-01 | NOTCH1 |
| Hypoxia | 2/200 | 2.49e-01 | TPBG;EGFR |
| Estrogen Response Late | 2/200 | 2.49e-01 | FOXC1;TPBG |
| Myogenesis | 2/200 | 2.49e-01 | NOTCH1;TCAP |
| p53 Pathway | 2/200 | 2.49e-01 | NOTCH1;KRT17 |
| Wnt-beta Catenin Signaling | 1/42 | 2.49e-01 | NOTCH1 |
| Apical Surface | 1/44 | 2.49e-01 | GATA3 |
| Reactive Oxygen Species Pathway | 1/49 | 2.57e-01 | G6PD |

## Biological takeaways

### High-confidence
- **LumA (± DE-pass):** Hallmark **E2F Targets**, **Mitotic Spindle**, **G2-M Checkpoint** (adj.P ~1e-4–2e-3); GO mitotic spindle / kinetochore organization (adj.P ~1e-5). Supports a **proliferation / cell-cycle** program (direction of DE: mitotic genes *lower* in LumA).
- **LumB:** same Hallmark direction but weaker (small gene list).
- **Inflammation:** not the dominant signal for luminal mitotic lists; a few HER2 GO interferon/inflammatory terms appear but are often single-gene and should not be over-claimed.

### Caution
- HER2 / Basal ORA hits can be driven by 1–2 genes after proteomics symbol mapping (GATA3, TSC2, MAP2K1, BCL2, KIT).
- Basal consensus is dominated by **17q12 amplicon symbols** — Enrichr will not always name this as “HER2 signaling”; interpret with DE + locus knowledge (and verify labels).

Full tables: `enrichr_*.csv` and `enrichment_all.csv` in this folder.
