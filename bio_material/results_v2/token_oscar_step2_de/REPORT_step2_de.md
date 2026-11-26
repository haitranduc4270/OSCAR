# Biomarker Step 2 — Differential Expression Report
## Method
- **Comparison:** one-vs-rest (subtype vs all others) on the full BRCA cohort (train∪val of one pre-split fold = all labeled samples).
- **Test:** Mann–Whitney U (two-sided).
- **Multiple testing:** Benjamini–Hochberg FDR within each **omic × class**.
- **Effect size:** Cohen's d and mean difference (matrices are already centered; classical log2FC on raw counts is not appropriate — `log2fc_pseudo` is only directional).
- **Pass criteria:** `q < 0.05` and `|Cohen's d| ≥ 0.3`.
- **Intersection:** step-1 consensus features (≥3/5 folds in gradient top-k).
## Cohort
- N samples (approx): **511**
- Class order: 0=LumA, 1=LumB, 2=Basal, 3=HER2

## Summary — consensus ∩ DE pass
| Subtype | Omic | Consensus | DE pass |
|---|---|---:|---:|
| Basal | CNV | 13 | 0 |
| Basal | Proteomics | 16 | 10 |
| Basal | mRNA | 11 | 9 |
| HER2 | CNV | 5 | 5 |
| HER2 | Proteomics | 15 | 11 |
| HER2 | mRNA | 6 | 6 |
| LumA | CNV | 2 | 0 |
| LumA | Proteomics | 12 | 7 |
| LumA | mRNA | 10 | 6 |
| LumB | CNV | 7 | 0 |
| LumB | Proteomics | 9 | 4 |
| LumB | mRNA | 3 | 3 |

**Total consensus:** 109 · **DE pass:** 61

## Per-subtype highlights

### LumA
| Feature | Omic | n_folds | mean\|grad\| | mean_diff | Cohen's d | q | Pass |
|---|---|---:|---:|---:|---:|---:|:---:|
| NDC80 | mRNA | 3 | 0.002679 | -1.431 | -2.042 | 1.95e-63 | yes |
| GTSE1 | mRNA | 3 | 0.002892 | -1.422 | -2.016 | 1.41e-60 | yes |
| DEPDC1B | mRNA | 3 | 0.003062 | -1.391 | -1.932 | 1.79e-58 | yes |
| TBC1D31 | mRNA | 4 | 0.003039 | -1.142 | -1.389 | 1.20e-39 | yes |
| ERALPHA | Proteomics | 4 | 0.004101 | 1.06 | 1.248 | 8.68e-28 | yes |
| S6 | Proteomics | 3 | 0.003434 | -0.8512 | -0.939 | 1.83e-24 | yes |
| P27_pT198 | Proteomics | 4 | 0.004987 | -0.677 | -0.718 | 1.19e-15 | yes |
| ERALPHA_pS118 | Proteomics | 4 | 0.003603 | 0.5445 | 0.565 | 7.26e-12 | yes |
| KRT14 | mRNA | 3 | 0.003242 | 0.4702 | 0.483 | 2.17e-09 | yes |
| RNASE7 | mRNA | 3 | 0.002564 | 0.4596 | 0.471 | 1.08e-12 | yes |
| CHK1_pS296 | Proteomics | 5 | 0.004043 | -0.3494 | -0.354 | 2.43e-05 | yes |
| G6PD | Proteomics | 3 | 0.004317 | -0.3297 | -0.334 | 1.59e-02 | yes |
| IGFBP2 | Proteomics | 3 | 0.004332 | 0.3281 | 0.332 | 3.44e-04 | yes |
| HER2_pY1248 | Proteomics | 5 | 0.004125 | -0.4505 | -0.461 | 2.01e-01 | no |
| HER2 | Proteomics | 5 | 0.004121 | -0.4174 | -0.426 | 1.01e-01 | no |
| P90RSK_pT359S363 | Proteomics | 3 | 0.003864 | 0.2826 | 0.285 | 1.99e-03 | no |
| KRT5 | mRNA | 4 | 0.002595 | 0.2232 | 0.224 | 2.83e-02 | no |
| DIP2A | CNV | 5 | 0.0004664 | -0.06015 | -0.199 | 3.92e-02 | no |
| KRT17 | mRNA | 4 | 0.002691 | 0.1758 | 0.176 | 1.82e-01 | no |
| DSG3 | mRNA | 3 | 0.002412 | 0.1237 | 0.124 | 6.01e-04 | no |
| KRT6B | mRNA | 4 | 0.002712 | -0.1005 | -0.100 | 4.56e-01 | no |
| ACC_pS79 | Proteomics | 3 | 0.003514 | -0.0434 | -0.043 | 9.44e-01 | no |
| AKT | Proteomics | 4 | 0.004231 | 0.03649 | 0.036 | 2.60e-01 | no |
| PXDNL | CNV | 3 | 0.0004732 | -0.0009199 | -0.002 | 9.92e-01 | no |

### LumB
| Feature | Omic | n_folds | mean\|grad\| | mean_diff | Cohen's d | q | Pass |
|---|---|---:|---:|---:|---:|---:|:---:|
| KRT5 | mRNA | 5 | 0.003619 | -1.068 | -1.175 | 2.75e-22 | yes |
| KRT17 | mRNA | 5 | 0.003845 | -1.055 | -1.157 | 6.63e-22 | yes |
| TBC1D31 | mRNA | 4 | 0.003552 | 0.8589 | 0.911 | 2.72e-14 | yes |
| GATA3 | Proteomics | 3 | 0.007219 | 0.8266 | 0.872 | 1.31e-12 | yes |
| ERALPHA | Proteomics | 5 | 0.0089 | 0.7641 | 0.800 | 3.41e-10 | yes |
| ERALPHA_pS118 | Proteomics | 3 | 0.006327 | 0.6483 | 0.669 | 4.36e-11 | yes |
| SHC_pY317 | Proteomics | 3 | 0.008055 | -0.6008 | -0.617 | 7.24e-10 | yes |
| EGFR_pY1068 | Proteomics | 4 | 0.007975 | -0.2461 | -0.247 | 5.75e-05 | no |
| HER2 | Proteomics | 4 | 0.008187 | 0.2385 | 0.239 | 3.27e-04 | no |
| ERK2 | Proteomics | 5 | 0.006926 | 0.08055 | 0.080 | 9.62e-01 | no |
| HER2_pY1248 | Proteomics | 4 | 0.008145 | 0.06262 | 0.063 | 2.65e-02 | no |
| PGAP3 | CNV | 3 | 0.0008144 | -0.02629 | -0.060 | 8.26e-01 | no |
| TCAP | CNV | 4 | 0.0008583 | -0.02387 | -0.055 | 8.26e-01 | no |
| STARD3 | CNV | 4 | 0.000851 | -0.02387 | -0.055 | 8.26e-01 | no |
| ERBB2 | CNV | 4 | 0.0008463 | -0.02387 | -0.054 | 8.26e-01 | no |
| PPP1R1B | CNV | 3 | 0.0006359 | -0.02145 | -0.049 | 8.26e-01 | no |
| GRB7 | CNV | 4 | 0.0008256 | -0.02145 | -0.049 | 8.26e-01 | no |
| P27_pT198 | Proteomics | 5 | 0.008136 | -0.03342 | -0.033 | 6.07e-01 | no |
| IKZF3 | CNV | 3 | 0.0008624 | -0.01418 | -0.032 | 8.26e-01 | no |

### Basal
| Feature | Omic | n_folds | mean\|grad\| | mean_diff | Cohen's d | q | Pass |
|---|---|---:|---:|---:|---:|---:|:---:|
| FOXC1 | mRNA | 3 | 0.0007696 | 1.996 | 3.533 | 4.98e-55 | yes |
| ERALPHA | Proteomics | 4 | 0.002687 | -1.658 | -2.273 | 5.19e-46 | yes |
| KRT6B | mRNA | 3 | 0.0006715 | 1.325 | 1.581 | 1.11e-33 | yes |
| PGAP3 | mRNA | 3 | 0.0005732 | -1.039 | -1.149 | 1.57e-29 | yes |
| ERALPHA_pS118 | Proteomics | 3 | 0.002289 | -0.98 | -1.070 | 9.11e-27 | yes |
| ERBB2 | mRNA | 3 | 0.0006155 | -0.9488 | -1.030 | 2.14e-24 | yes |
| ROS1 | mRNA | 3 | 0.0007059 | 0.8598 | 0.918 | 7.85e-14 | yes |
| CKIT | Proteomics | 4 | 0.003309 | 0.8292 | 0.881 | 1.87e-09 | yes |
| PNMT | mRNA | 3 | 0.0006362 | -0.77 | -0.811 | 2.81e-14 | yes |
| STARD3 | mRNA | 4 | 0.000598 | -0.6916 | -0.720 | 3.26e-12 | yes |
| HER2 | Proteomics | 5 | 0.003982 | -0.683 | -0.711 | 3.29e-13 | yes |
| P27_pT198 | Proteomics | 5 | 0.002751 | 0.6439 | 0.667 | 6.75e-09 | yes |
| VEGFR2 | Proteomics | 3 | 0.002419 | -0.5547 | -0.569 | 4.46e-07 | yes |
| MIEN1 | mRNA | 3 | 0.000415 | -0.5241 | -0.536 | 3.97e-07 | yes |
| ACC_pS79 | Proteomics | 4 | 0.002705 | -0.5141 | -0.525 | 1.68e-06 | yes |
| HER2_pY1248 | Proteomics | 5 | 0.003727 | -0.5107 | -0.522 | 5.05e-11 | yes |
| TCAP | mRNA | 3 | 0.0003986 | -0.4215 | -0.427 | 6.04e-04 | yes |
| ARAF_pS299 | Proteomics | 4 | 0.002584 | 0.4195 | 0.425 | 6.02e-05 | yes |
| ERK2 | Proteomics | 3 | 0.002476 | -0.3607 | -0.364 | 2.81e-02 | yes |
| GRB7 | mRNA | 4 | 0.0003768 | -0.2882 | -0.290 | 7.20e-02 | no |
| STAT3_pY705 | Proteomics | 3 | 0.002792 | -0.2799 | -0.281 | 8.55e-02 | no |
| G6PD | Proteomics | 5 | 0.003223 | -0.251 | -0.252 | 2.51e-02 | no |
| S6_pS235S236 | Proteomics | 4 | 0.002815 | 0.2501 | 0.251 | 1.54e-02 | no |
| EGFR_pY1068 | Proteomics | 5 | 0.003089 | -0.2147 | -0.215 | 6.06e-02 | no |
| HER3_pY1289 | Proteomics | 4 | 0.002751 | -0.1831 | -0.183 | 6.06e-02 | no |
| ZPBP2 | CNV | 4 | 0.0003502 | -0.06062 | -0.142 | 4.13e-01 | no |
| IKZF3 | CNV | 4 | 0.0003482 | -0.05388 | -0.122 | 4.13e-01 | no |
| GRB7 | CNV | 4 | 0.0003989 | -0.04997 | -0.114 | 4.13e-01 | no |
| CACNA1G | CNV | 3 | 0.0003888 | -0.04323 | -0.114 | 4.13e-01 | no |
| GSDMB | CNV | 3 | 0.0004246 | -0.04417 | -0.105 | 4.13e-01 | no |
| CSF3 | CNV | 3 | 0.0004878 | -0.04308 | -0.104 | 4.13e-01 | no |
| PGAP3 | CNV | 4 | 0.0003705 | -0.04355 | -0.100 | 4.13e-01 | no |
| TCAP | CNV | 4 | 0.000395 | -0.04104 | -0.094 | 4.13e-01 | no |
| STARD3 | CNV | 4 | 0.0003913 | -0.04104 | -0.094 | 4.13e-01 | no |
| PNMT | CNV | 3 | 0.0001858 | -0.04104 | -0.094 | 4.13e-01 | no |
| ERBB2 | CNV | 4 | 0.0004027 | -0.04104 | -0.093 | 4.13e-01 | no |
| PPP1R1B | CNV | 4 | 0.0003753 | -0.03853 | -0.089 | 4.13e-01 | no |
| CDK12 | CNV | 4 | 0.0003581 | -0.02099 | -0.048 | 6.17e-01 | no |
| CD49B | Proteomics | 3 | 0.002299 | 0.03496 | 0.035 | 6.42e-01 | no |
| RTL4 | mRNA | 3 | 0.0005414 | 0.008157 | 0.008 | 6.21e-01 | no |

### HER2
| Feature | Omic | n_folds | mean\|grad\| | mean_diff | Cohen's d | q | Pass |
|---|---|---:|---:|---:|---:|---:|:---:|
| HER2_pY1248 | Proteomics | 4 | 0.007554 | 2.046 | 2.613 | 3.85e-16 | yes |
| HER2 | Proteomics | 4 | 0.006667 | 1.981 | 2.481 | 6.58e-16 | yes |
| TCAP | mRNA | 3 | 0.001422 | 1.815 | 2.175 | 6.50e-20 | yes |
| GRB7 | mRNA | 4 | 0.001707 | 1.802 | 2.153 | 9.66e-18 | yes |
| ERBB2 | mRNA | 4 | 0.001859 | 1.782 | 2.119 | 1.13e-18 | yes |
| GRB7 | CNV | 4 | 0.0005272 | 0.7299 | 1.935 | 1.24e-31 | yes |
| TCAP | CNV | 3 | 0.0006762 | 0.7067 | 1.869 | 1.51e-30 | yes |
| STARD3 | CNV | 3 | 0.0006322 | 0.7067 | 1.869 | 1.51e-30 | yes |
| IKZF3 | CNV | 3 | 0.0005641 | 0.6944 | 1.791 | 4.85e-29 | yes |
| EGFR_pY1068 | Proteomics | 3 | 0.005253 | 1.554 | 1.762 | 7.18e-15 | yes |
| G6PD | Proteomics | 4 | 0.006491 | 1.512 | 1.700 | 7.40e-15 | yes |
| CDK12 | CNV | 3 | 0.000326 | 0.6189 | 1.580 | 3.90e-24 | yes |
| ESR1 | mRNA | 3 | 0.001351 | -1.091 | -1.154 | 1.03e-12 | yes |
| ERALPHA | Proteomics | 4 | 0.005705 | -1.071 | -1.131 | 6.69e-12 | yes |
| CCDC6 | mRNA | 3 | 0.001503 | 0.9682 | 1.011 | 1.10e-10 | yes |
| ARAF_pS299 | Proteomics | 3 | 0.007113 | 0.8076 | 0.832 | 1.81e-05 | yes |
| TPBG | mRNA | 3 | 0.001847 | -0.7556 | -0.775 | 2.61e-09 | yes |
| ERALPHA_pS118 | Proteomics | 4 | 0.005179 | -0.7399 | -0.758 | 9.97e-08 | yes |
| IRS1 | Proteomics | 3 | 0.005279 | -0.626 | -0.637 | 5.69e-07 | yes |
| GATA3 | Proteomics | 5 | 0.006634 | -0.5722 | -0.580 | 3.59e-05 | yes |
| CKIT | Proteomics | 3 | 0.00455 | -0.5346 | -0.541 | 2.36e-05 | yes |
| RB_pS807S811 | Proteomics | 3 | 0.004137 | 0.3949 | 0.397 | 1.93e-02 | yes |
| AR | Proteomics | 5 | 0.004882 | 0.329 | 0.330 | 8.05e-02 | no |
| NOTCH1 | Proteomics | 4 | 0.00396 | -0.1057 | -0.106 | 9.80e-01 | no |
| AMPKALPHA_pT172 | Proteomics | 4 | 0.005594 | 0.06404 | 0.064 | 9.80e-01 | no |
| CHK2 | Proteomics | 3 | 0.004345 | -0.01863 | -0.019 | 7.24e-01 | no |

## Interpretation notes
- Features with **de_pass=yes** are supported both by model sensitivity (gradient) and by univariate subtype difference.
- **Sign of mean_diff / Cohen's d:** positive → higher in the subtype vs rest (on the processed scale); negative → lower.
- Failures (consensus but not DE) may still matter to the model via multivariate interactions; treat them as model-facing, not classical biomarkers.
- Basal 17q12 CNV block: check whether DE direction/significance matches HER2 biology; if amplicon genes DE for Basal unexpectedly, revisit label encoding.
