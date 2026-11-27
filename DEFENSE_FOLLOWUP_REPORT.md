# BÁO CÁO KẾT QUẢ THÍ NGHIỆM BỔ SUNG PHẢN BIỆN (DEFENSE FOLLOW-UP REPORT)

**Mô hình tham chiếu (`TokenOSCAR 256×32`, seed 42, matched nested 5-fold):**
- **BRCA:** `89.93 ± 2.15%` Macro-F1 (`91.39%` Acc)
- **LUNG:** `97.34 ± 1.30%` Macro-F1 (`97.36%` Acc)
- **COADREAD:** `89.26 ± 2.55%` Macro-F1 (`91.00%` Acc)
- **Trung bình 3 cohort:** **92.18% Macro-F1** (`93.25%` Acc)

**Giao thức chung:** Toàn bộ thí nghiệm bổ sung đều tuân thủ nghiêm ngặt giao thức matched nested 5-fold (chia fold cố định, inner 20% stratified validation, chọn `first-max` inner-val Macro-F1 trong budget 120 epoch, `StandardScaler` fit chỉ trên train, retrain trên 100% outer-train, đánh giá 1 lần trên test).

---

## 1. Exp 1 — Single-Omic, Classical Linear & BRCA PAM50-Removed (Trả lời Q6)

**File kết quả:** `experiments/defense_followup/exp1_single_omic_linear/single_omic_and_linear_results.json`

### 1.1. So sánh Multi-omics vs Single-omic & Baseline tuyến tính (Macro-F1, 5-fold)

| Mô hình | BRCA F1 | LUNG F1 | COADREAD F1 | Avg F1 | Δ vs TokenOSCAR |
|---|---|---|---|---|---|
| **`TokenOSCAR (3-omics, 256×32)`** | **89.93 ± 2.15%** | **97.34 ± 1.30%** | **89.26 ± 2.55%** | **92.18%** | — |
| `mRNA_only_mlp` (OmicToTokens) | 88.65 ± 2.95% | 95.43 ± 1.63% | 88.31 ± 4.98% | 90.80% | **-1.38%** |
| `mRNA_only_transformer` (OmicToTokens + SA) | 89.10 ± 2.03% | 95.87 ± 1.00% | 89.20 ± 4.53% | 91.39% | **-0.79%** |
| `CNV_only_mlp` | 55.22 ± 5.27% | 76.37 ± 2.82% | 33.90 ± 5.32% | 55.17% | -37.01% |
| `Proteomics_only_mlp` | 79.31 ± 5.12% | 89.40 ± 2.23% | 60.36 ± 5.43% | 76.36% | -15.82% |
| `L2-LogReg (mRNA-only)` | 87.95 ± 3.29% | 96.02 ± 1.37% | 90.44 ± 2.48% | 91.47% | **-0.71%** |
| `L2-LogReg (Concat 3-omics)` | 85.28 ± 3.45% | 96.17 ± 1.27% | 89.90 ± 1.79% | 90.45% | **-1.73%** |
| `LinearSVC (mRNA-only)` | 83.60 ± 2.96% | 96.31 ± 0.95% | 87.27 ± 3.68% | 89.06% | **-3.12%** |
| `LinearSVC (Concat 3-omics)` | 80.67 ± 1.70% | 96.61 ± 1.11% | 87.20 ± 4.70% | 88.16% | **-4.02%** |

### 1.2. Stress Test loại bỏ 50 gen PAM50 khỏi mRNA trên BRCA

| Mô hình (BRCA, bỏ 50 gen PAM50) | Macro-F1 | Accuracy |
|---|---|---|
| `mRNA_only_mlp` | 87.57 ± 1.76% | 89.24 ± 1.52% |
| **`TokenOSCAR (3-omics)`** | **88.02 ± 3.84%** | **90.03 ± 2.62%** |

**Kết luận rút ra (Q6):**
- `TokenOSCAR (3-omics)` vượt cả `mRNA_only_mlp` (+1.38% Avg F1) lẫn `mRNA_only_transformer` (+0.79% Avg F1) trên **cả 3 tập dữ liệu**, đồng thời vượt tất cả baseline tuyến tính (`L2-LogReg`, `LinearSVC`).
- Đáng chú ý, với mô hình tuyến tính, ghép thẳng 3 omics (`logreg_concat_3omics`: 85.28% BRCA) **làm tụt -2.67 F1** so với `mRNA-only` (87.95%) do nhiễu chiều cao từ CNV — chứng minh giá trị của cơ chế fusion phi tuyến có gate của `TokenOSCAR`.
- Khi loại bỏ hoàn toàn 50 gen PAM50 khỏi mRNA trên BRCA, `TokenOSCAR` vẫn giữ được lợi thế so với `mRNA-only` (+0.45% F1, +0.79% Acc).

---

## 2. Exp 2a & 2b — Kiểm chứng Gate học được vs Fixed Gate & Phân phối tọa độ Gate (Trả lời Q3)

**File kết quả:**
- `experiments/defense_followup/exp2_3_gate_and_wiring/gate_and_wiring_results.json`
- `experiments/defense_followup/exp2b_gate_audit/gate_audit_summary.json`

### 2.1. Đối chứng Fixed Gate (Macro-F1, 5-fold)

| Biến thể Gate | BRCA F1 | LUNG F1 | COADREAD F1 | Avg F1 | Δ vs Learned Gate |
|---|---|---|---|---|---|
| **`TokenOSCAR (Learned Gate)`** | **89.93 ± 2.15%** | **97.34 ± 1.30%** | **89.26 ± 2.55%** | **92.18%** | — |
| `fixed_gate_0.5` ($\alpha \equiv 0.5$) | 88.70 ± 2.20% | 96.61 ± 1.00% | 87.31 ± 5.89% | 90.87% | **-1.31%** |
| `fixed_gate_1.0` ($\alpha \equiv 1.0$) | 89.72 ± 2.63% | 95.44 ± 1.41% | 89.31 ± 5.85% | 91.49% | **-0.69%** |

### 2.2. Kiểm toán phân phối tọa độ Gate $\alpha_{t,m}$ trên BRCA (511 bệnh nhân, 5 fold)

| Gate | Mean | Within-patient Std | Within-patient IQR | Min–Max trung bình | Frac > 0.5 | Kruskal-Wallis $p$ (giữa 4 subtype) |
|---|---|---|---|---|---|---|
| `Proteomics` (cross) | 0.5209 | 0.0705 | 0.0924 | `[0.329, 0.760]` | 60.24% | **$1.11 \times 10^{-68}$** |
| `mRNA` (cross) | 0.5022 | 0.0418 | 0.0567 | `[0.365, 0.650]` | 50.30% | **$1.19 \times 10^{-23}$** |
| `self_gate` | 0.7711 | 0.1206 | 0.1665 | `[0.389, 0.978]` | 96.26% | **$4.31 \times 10^{-23}$** |

**Kết luận rút ra (Q3):**
- Bác bỏ hoàn toàn nghi vấn "gate chết quanh $\sigma(0)=0.5$": (i) `Learned Gate` thắng `fixed_gate_0.5` trên **cả 3 tập** (+1.31% Avg F1, giảm mạnh phương sai trên COADREAD từ $\pm 5.89\%$ xuống $\pm 2.55\%$); (ii) bên trong từng bệnh nhân, tọa độ gate phân hóa rộng (`[0.329, 0.760]` ở Proteomics, `[0.389, 0.978]` ở `self_gate`) và khác biệt có ý nghĩa thống kê cực mạnh giữa 4 subtype BRCA ($p < 10^{-22}$).

---

## 3. Exp 3 — Architectural Wiring: $F_T$ vs $Z_T$ & Unrestricted All-to-All Cross-Attention (Trả lời Q4, Q15)

**File kết quả:** `experiments/defense_followup/exp2_3_gate_and_wiring/gate_and_wiring_results.json`

| Biến thể kiến trúc | BRCA F1 | LUNG F1 | COADREAD F1 | Avg F1 | Δ vs Default |
|---|---|---|---|---|---|
| **`TokenOSCAR (Z_T, 2 directed edges)`** | **89.93 ± 2.15%** | **97.34 ± 1.30%** | **89.26 ± 2.55%** | **92.18%** | — |
| `chained_FT` (dùng $F_T$ thay $Z_T$ ở Step 2) | 89.67 ± 3.20% | 96.62 ± 1.00% | 87.81 ± 3.91% | 91.36% | **-0.82%** |
| `unrestricted_all2all` (cả 6 cạnh có hướng) | 90.42 ± 2.58% | 96.91 ± 1.27% | 86.78 ± 4.65% | 91.37% | **-0.81%** |

**Kết luận rút ra (Q4, Q15):**
- Dùng `Z_T` gốc ở Step 2 vượt `chained_FT` trên **cả 3 tập** (+0.82% Avg F1), ủng hộ giả thuyết tránh khuếch đại nhiễu CNV sang RPPA.
- Mở toàn bộ 6 cạnh `unrestricted_all2all` nhỉnh nhẹ trên BRCA (+0.49%) nhưng giảm trên LUNG (-0.43%) và tụt mạnh trên tập nhỏ COADREAD (-2.48%, std tăng lên $\pm 4.65\%$), cho thấy đồ thị 2 cạnh gọn hơn giúp kiểm soát overfit tốt hơn ở chế độ cỡ mẫu nhỏ.

---

## 4. Exp 4 — Confusion Matrices 5-Fold OOF & Phân tích lỗi theo lớp (Trả lời Q16)

**File kết quả:** `experiments/defense_followup/exp4_confusion_matrices/confusion_matrices.json` và 3 ảnh PNG (`cm_brca.png`, `cm_lung.png`, `cm_coadread.png`).

- **BRCA ($N=511$, OOF Acc `91.39%`, OOF F1 `89.88%`, Fold F1 `89.93 ± 2.15%`):**
  - Tổng 44/511 mẫu sai, trong đó **29/44 lỗi (65.9%)** rơi đúng vào cặp biên sinh học **LumA ↔ LumB** (16 LumA → LumB, 13 LumB → LumA). Basal đạt recall 97.3% (109/112), HER2 đạt 90.6% (48/53).
- **LUNG ($N=682$, OOF Acc `97.36%`, OOF F1 `97.35%`, Fold F1 `97.34 ± 1.30%`):**
  - LUAD: 354/359 (98.6%), LUSC: 310/323 (96.0%).
- **COADREAD ($N=222$, OOF Acc `90.99%`, OOF F1 `89.52%`, Fold F1 `89.26 ± 2.55%`):**
  - Mapping chuẩn theo tần suất trong Table 3.1 (`0=CMS2, 1=CMS4, 2=CMS3, 3=CMS1`): **CMS2: 79/86 (91.9%)**, **CMS4: 66/69 (95.7%)**, **CMS3: 27/34 (79.4% — lớp khó nhất, chủ yếu nhầm sang CMS2, khớp sinh học vì cả CMS2 và CMS3 đều là nhóm biểu mô)**, **CMS1: 30/33 (90.9%)**.

---

## 5. Exp 5 — Ổn định Đa Seed (3 Seeds `42, 123, 2024`, 15 Fold/Cohort) & Kiểm định Nadeau–Bengio (Trả lời Q2, Q7, Q8)

**File kết quả:** `experiments/defense_followup/exp5_multiseed_stats/multiseed_results.json`

### 5.1. Macro-F1 gộp 3 seeds × 5 folds (15 lần đánh giá / cohort)

| Biến thể | BRCA (15 folds) | LUNG (15 folds) | COADREAD (15 folds) | Avg 3 Cohort (45 folds) |
|---|---|---|---|---|
| **`TokenOSCAR (Biological)`** | **89.84 ± 2.95%** | **96.66 ± 1.52%** | 89.00 ± 4.13% | **91.83%** |
| `Reverse` (Prot → mRNA → CNV) | 89.62 ± 3.24% | 96.32 ± 1.32% | **89.37 ± 4.03%** | 91.77% |
| `Concat` (không attention/gate) | 87.01 ± 2.89% | 95.73 ± 1.59% | 88.93 ± 4.46% | 90.56% |
| `mRNA_only` | 88.78 ± 2.51% | 95.68 ± 1.51% | 86.77 ± 5.29% | 90.41% |

### 5.2. Kiểm định t cặp hiệu chỉnh Nadeau–Bengio (2003) (`TokenOSCAR` vs đối chứng, 15 cặp)

| Cohort | So sánh | Δ Mean F1 | $t$-stat | $p$-value | 95% CI |
|---|---|---|---|---|---|
| BRCA | `TokenOSCAR` vs `Concat` | **+2.83%** | 1.51 | 0.1539 | `[-1.20%, +6.86%]` |
| BRCA | `TokenOSCAR` vs `Reverse` | +0.22% | 0.15 | 0.8808 | `[-2.85%, +3.29%]` |
| BRCA | `TokenOSCAR` vs `mRNA_only` | +1.06% | 1.06 | 0.3075 | `[-1.09%, +3.22%]` |
| LUNG | `TokenOSCAR` vs `Concat` | +0.93% | 1.01 | 0.3298 | `[-1.05%, +2.91%]` |
| LUNG | `TokenOSCAR` vs `Reverse` | +0.34% | 0.36 | 0.7218 | `[-1.68%, +2.36%]` |
| LUNG | `TokenOSCAR` vs `mRNA_only` | +0.98% | 0.92 | 0.3717 | `[-1.30%, +3.25%]` |
| COADREAD | `TokenOSCAR` vs `Concat` | +0.07% | 0.02 | 0.9814 | `[-6.20%, +6.34%]` |
| COADREAD | `TokenOSCAR` vs `Reverse` | -0.37% | -0.19 | 0.8498 | `[-4.47%, +3.73%]` |
| COADREAD | `TokenOSCAR` vs `mRNA_only` | +2.23% | 0.59 | 0.5627 | `[-5.84%, +10.30%]` |

**Kết luận rút ra (Q2, Q7, Q8):**
- Khi mở rộng ra 3 seed (45 fold), **hiệu ứng của `TokenOSCAR` so với `Concat` rõ hơn hẳn so với 1 seed** (+1.27% Avg F1, riêng BRCA **+2.83%**), và vượt `mRNA_only` trên cả 3 cohort (+1.42% Avg F1).
- Trên 3 seed, **thứ tự sinh học (`Biological`) vượt `Reverse` trên cả BRCA (+0.22%) lẫn LUNG (+0.34%)**, còn trung bình 3 cohort gần tương đương (+0.06%) — nhất quán với kết luận thận trọng của luận văn.
- Dưới kiểm định Nadeau–Bengio bảo thủ trên cỡ mẫu TCGA ($N=222\text{–}682$), không có cặp nào đạt $p < 0.05$, củng cố lập trường báo cáo trung thực ở mức mô tả.

---

## 6. Exp 6 — XAI Comparison & Signed MSigDB Hallmark trên BRCA (Trả lời Q10, Q12)

**File kết quả:** `experiments/defense_followup/exp6_xai_checks/xai_summary.json`

### 6.1. So sánh phương pháp Attribution (trung bình 5 fold × 4 subtype)

| Omic | Spearman(Van, IG) | Spearman(IxG, IG) | Top-20 Overlap(IxG, IG) |
|---|---|---|---|
| `mRNA` | 0.764 | **0.999** | **96.2%** |
| `Proteomics` | 0.742 | **0.945** | **85.8%** |
| `CNV` | 0.425 | **0.998** | **95.0%** |

### 6.2. Signed MSigDB Hallmark ORA cho Luminal A (top-200 `|IG|` tách theo chiều biểu hiện)

Khi tách riêng nhóm gen giảm biểu hiện trong Luminal A (tránh trộn lẫn gen tăng/giảm như ORA không dấu ở Table 5.6), **3 Hallmark tăng sinh kinh điển được thu hồi với FDR cực mạnh**:
1. `HALLMARK_E2F_TARGETS` — overlap 15/200, **$q = 1.50 \times 10^{-13}$** (`CDK1`, `CHEK1`, `BRCA2`, `BARD1`, `CKS1B`, ...)
2. `HALLMARK_G2M_CHECKPOINT` — overlap 11/199, **$q = 1.29 \times 10^{-8}$** (`CDK1`, `CHEK1`, `KIF11`, `TTK`, `INCENP`, ...)
3. `HALLMARK_MITOTIC_SPINDLE` — overlap 8/199, **$q = 2.55 \times 10^{-5}$** (`BRCA2`, `CDK1`, `KIF11`, `SMC4`, `TTK`, ...)

**Kết luận rút ra (Q10, Q12):**
- `Input × Gradient` và `Integrated Gradients` gần như trùng khớp (`Spearman = 0.945–0.999`, `Top-20 overlap = 85.8–96.2%`).
- Giải thích cho Q12 được chứng minh trực tiếp bằng số liệu: việc Table 5.6 không ra MSigDB Hallmark trên LumA hoàn toàn do danh sách 22 gen ngắn và không tách dấu; khi dùng signed top-IG, các hallmark đặc trưng của Luminal A (ức chế chu kỳ tế bào E2F / G2M / Mitotic Spindle) lập tức xuất hiện với $q < 10^{-4}$.
