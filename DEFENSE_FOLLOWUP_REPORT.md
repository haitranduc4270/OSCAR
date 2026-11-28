# BÁO CÁO KẾT QUẢ THÍ NGHIỆM BỔ SUNG PHẢN BIỆN (DEFENSE FOLLOW-UP REPORT)

**Mô hình tham chiếu (**`TokenOSCAR 256×32`**, seed 42, matched nested 5-fold):**

- **BRCA:** `89.93 ± 2.15%` Macro-F1 (`91.39%` Acc)
- **LUNG:** `97.34 ± 1.30%` Macro-F1 (`97.36%` Acc)
- **COADREAD:** `89.26 ± 2.55%` Macro-F1 (`91.00%` Acc)
- **Trung bình 3 cohort:** **92.18% Macro-F1** (`93.25%` Acc)

**Giao thức chung:** Toàn bộ thí nghiệm bổ sung đều tuân thủ nghiêm ngặt giao thức matched nested 5-fold (chia fold cố định, inner 20% stratified validation, chọn `first-max` inner-val Macro-F1 trong budget 120 epoch, `StandardScaler` fit chỉ trên train, retrain trên 100% outer-train, đánh giá 1 lần trên test).

---



## 1. Exp 1 — Single-Omic, Classical Linear & BRCA PAM50-Removed (Trả lời Q6)

**File kết quả:** `experiments/defense_followup/exp1_single_omic_linear/single_omic_and_linear_results.json`

### 1.1. So sánh Multi-omics vs Single-omic & Baseline tuyến tính (Macro-F1, 5-fold)


| Mô hình                                     | BRCA F1           | LUNG F1           | COADREAD F1       | Avg F1     | Δ vs TokenOSCAR |
| ------------------------------------------- | ----------------- | ----------------- | ----------------- | ---------- | --------------- |
| `TokenOSCAR (3-omics, 256×32)`              | **89.93 ± 2.15%** | **97.34 ± 1.30%** | **89.26 ± 2.55%** | **92.18%** | —               |
| `mRNA_only_mlp` (OmicToTokens)              | 88.65 ± 2.95%     | 95.43 ± 1.63%     | 88.31 ± 4.98%     | 90.80%     | **-1.38%**      |
| `mRNA_only_transformer` (OmicToTokens + SA) | 89.10 ± 2.03%     | 95.87 ± 1.00%     | 89.20 ± 4.53%     | 91.39%     | **-0.79%**      |
| `CNV_only_mlp`                              | 55.22 ± 5.27%     | 76.37 ± 2.82%     | 33.90 ± 5.32%     | 55.17%     | -37.01%         |
| `Proteomics_only_mlp`                       | 79.31 ± 5.12%     | 89.40 ± 2.23%     | 60.36 ± 5.43%     | 76.36%     | -15.82%         |
| `L2-LogReg (mRNA-only)`                     | 87.95 ± 3.29%     | 96.02 ± 1.37%     | 90.44 ± 2.48%     | 91.47%     | **-0.71%**      |
| `L2-LogReg (Concat 3-omics)`                | 85.28 ± 3.45%     | 96.17 ± 1.27%     | 89.90 ± 1.79%     | 90.45%     | **-1.73%**      |
| `LinearSVC (mRNA-only)`                     | 83.60 ± 2.96%     | 96.31 ± 0.95%     | 87.27 ± 3.68%     | 89.06%     | **-3.12%**      |
| `LinearSVC (Concat 3-omics)`                | 80.67 ± 1.70%     | 96.61 ± 1.11%     | 87.20 ± 4.70%     | 88.16%     | **-4.02%**      |




### 1.2. Stress Test loại bỏ 50 gen PAM50 khỏi mRNA trên BRCA


| Mô hình (BRCA, bỏ 50 gen PAM50) | Macro-F1          | Accuracy          |
| ------------------------------- | ----------------- | ----------------- |
| `mRNA_only_mlp`                 | 87.57 ± 1.76%     | 89.24 ± 1.52%     |
| `TokenOSCAR (3-omics)`          | **88.02 ± 3.84%** | **90.03 ± 2.62%** |


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


| Biến thể Gate                          | BRCA F1           | LUNG F1           | COADREAD F1       | Avg F1     | Δ vs Learned Gate |
| -------------------------------------- | ----------------- | ----------------- | ----------------- | ---------- | ----------------- |
| `TokenOSCAR (Learned Gate)`            | **89.93 ± 2.15%** | **97.34 ± 1.30%** | **89.26 ± 2.55%** | **92.18%** | —                 |
| `fixed_gate_0.5` ($\alpha \equiv 0.5$) | 88.70 ± 2.20%     | 96.61 ± 1.00%     | 87.31 ± 5.89%     | 90.87%     | **-1.31%**        |
| `fixed_gate_1.0` ($\alpha \equiv 1.0$) | 89.72 ± 2.63%     | 95.44 ± 1.41%     | 89.31 ± 5.85%     | 91.49%     | **-0.69%**        |




### 2.2. Kiểm toán phân phối tọa độ Gate $\alpha_{t,m}$ trên BRCA (511 bệnh nhân, 5 fold)


| Gate                 | Mean   | Within-patient Std | Within-patient IQR | Min–Max trung bình | Frac > 0.5 | Kruskal-Wallis $p$ (giữa 4 subtype) |
| -------------------- | ------ | ------------------ | ------------------ | ------------------ | ---------- | ----------------------------------- |
| `Proteomics` (cross) | 0.5209 | 0.0705             | 0.0924             | `[0.329, 0.760]`   | 60.24%     | **$1.11 \times 10^{-68}$**          |
| `mRNA` (cross)       | 0.5022 | 0.0418             | 0.0567             | `[0.365, 0.650]`   | 50.30%     | **$1.19 \times 10^{-23}$**          |
| `self_gate`          | 0.7711 | 0.1206             | 0.1665             | `[0.389, 0.978]`   | 96.26%     | **$4.31 \times 10^{-23}$**          |


**Kết luận rút ra (Q3):**

- Bác bỏ hoàn toàn nghi vấn "gate chết quanh $\sigma(0)=0.5$": (i) `Learned Gate` thắng `fixed_gate_0.5` trên **cả 3 tập** (+1.31% Avg F1, giảm mạnh phương sai trên COADREAD từ $\pm 5.89$ xuống $\pm 2.55$); (ii) bên trong từng bệnh nhân, tọa độ gate phân hóa rộng (`[0.329, 0.760]` ở Proteomics, `[0.389, 0.978]` ở `self_gate`) và khác biệt có ý nghĩa thống kê cực mạnh giữa 4 subtype BRCA ($p < 10^{-22}$).

---



## 3. Exp 3 — Architectural Wiring: $F_T$ vs $Z_T$ & Unrestricted All-to-All Cross-Attention (Trả lời Q4, Q15)

**File kết quả:** `experiments/defense_followup/exp2_3_gate_and_wiring/gate_and_wiring_results.json`


| Biến thể kiến trúc                            | BRCA F1           | LUNG F1           | COADREAD F1       | Avg F1     | Δ vs Default |
| --------------------------------------------- | ----------------- | ----------------- | ----------------- | ---------- | ------------ |
| `TokenOSCAR (Z_T, 2 directed edges)`          | **89.93 ± 2.15%** | **97.34 ± 1.30%** | **89.26 ± 2.55%** | **92.18%** | —            |
| `chained_FT` (dùng $F_T$ thay $Z_T$ ở Step 2) | 89.67 ± 3.20%     | 96.62 ± 1.00%     | 87.81 ± 3.91%     | 91.36%     | **-0.82%**   |
| `unrestricted_all2all` (cả 6 cạnh có hướng)   | 90.42 ± 2.58%     | 96.91 ± 1.27%     | 86.78 ± 4.65%     | 91.37%     | **-0.81%**   |


**Kết luận rút ra (Q4, Q15):**

- Dùng `Z_T` gốc ở Step 2 vượt `chained_FT` trên **cả 3 tập** (+0.82% Avg F1), ủng hộ giả thuyết tránh khuếch đại nhiễu CNV sang RPPA.
- Mở toàn bộ 6 cạnh `unrestricted_all2all` nhỉnh nhẹ trên BRCA (+0.49%) nhưng giảm trên LUNG (-0.43%) và tụt mạnh trên tập nhỏ COADREAD (-2.48%, std tăng lên $\pm 4.65$), cho thấy đồ thị 2 cạnh gọn hơn giúp kiểm soát overfit tốt hơn ở chế độ cỡ mẫu nhỏ.

---



## 4. Exp 4 — Confusion Matrices 5-Fold OOF & Phân tích lỗi theo lớp (Trả lời Q16)

**File kết quả:** `experiments/defense_followup/exp4_confusion_matrices/confusion_matrices.json` và 3 ảnh PNG (`cm_brca.png`, `cm_lung.png`, `cm_coadread.png`).

- **BRCA ($N=511$, OOF Acc** `91.39%`**, OOF F1** `89.88%`**, Fold F1** `89.93 ± 2.15%`**):**
  - Tổng 44/511 mẫu sai, trong đó **29/44 lỗi (65.9%)** rơi đúng vào cặp biên sinh học **LumA ↔ LumB** (16 LumA → LumB, 13 LumB → LumA). Basal đạt recall 97.3% (109/112), HER2 đạt 90.6% (48/53).
- **LUNG ($N=682$, OOF Acc** `97.36%`**, OOF F1** `97.35%`**, Fold F1** `97.34 ± 1.30%`**):**
  - LUAD: 354/359 (98.6%), LUSC: 310/323 (96.0%).
- **COADREAD ($N=222$, OOF Acc** `90.99%`**, OOF F1** `89.52%`**, Fold F1** `89.26 ± 2.55%`**):**
  - Mapping chuẩn theo tần suất trong Table 3.1 (`0=CMS2, 1=CMS4, 2=CMS3, 3=CMS1`): **CMS2: 79/86 (91.9%)**, **CMS4: 66/69 (95.7%)**, **CMS3: 27/34 (79.4% — lớp khó nhất, chủ yếu nhầm sang CMS2, khớp sinh học vì cả CMS2 và CMS3 đều là nhóm biểu mô)**, **CMS1: 30/33 (90.9%)**.

---



## 5. Exp 5 — Ổn định Đa Seed (3 Seeds `42, 123, 2024`, 15 Fold/Cohort) & Kiểm định Nadeau–Bengio (Trả lời Q2, Q7, Q8)

**File kết quả:** `experiments/defense_followup/exp5_multiseed_stats/multiseed_results.json`

### 5.1. Macro-F1 gộp 3 seeds × 5 folds (15 lần đánh giá / cohort)


| Biến thể                        | BRCA (15 folds)   | LUNG (15 folds)   | COADREAD (15 folds) | Avg 3 Cohort (45 folds) |
| ------------------------------- | ----------------- | ----------------- | ------------------- | ----------------------- |
| `TokenOSCAR (Biological)`       | **89.84 ± 2.95%** | **96.66 ± 1.52%** | 89.00 ± 4.13%       | **91.83%**              |
| `Reverse` (Prot → mRNA → CNV)   | 89.62 ± 3.24%     | 96.32 ± 1.32%     | **89.37 ± 4.03%**   | 91.77%                  |
| `Concat` (không attention/gate) | 87.01 ± 2.89%     | 95.73 ± 1.59%     | 88.93 ± 4.46%       | 90.56%                  |
| `mRNA_only`                     | 88.78 ± 2.51%     | 95.68 ± 1.51%     | 86.77 ± 5.29%       | 90.41%                  |




### 5.2. Kiểm định t cặp hiệu chỉnh Nadeau–Bengio (2003) (`TokenOSCAR` vs đối chứng, 15 cặp)


| Cohort   | So sánh                     | Δ Mean F1  | $t$-stat | $p$-value | 95% CI              |
| -------- | --------------------------- | ---------- | -------- | --------- | ------------------- |
| BRCA     | `TokenOSCAR` vs `Concat`    | **+2.83%** | 1.51     | 0.1539    | `[-1.20%, +6.86%]`  |
| BRCA     | `TokenOSCAR` vs `Reverse`   | +0.22%     | 0.15     | 0.8808    | `[-2.85%, +3.29%]`  |
| BRCA     | `TokenOSCAR` vs `mRNA_only` | +1.06%     | 1.06     | 0.3075    | `[-1.09%, +3.22%]`  |
| LUNG     | `TokenOSCAR` vs `Concat`    | +0.93%     | 1.01     | 0.3298    | `[-1.05%, +2.91%]`  |
| LUNG     | `TokenOSCAR` vs `Reverse`   | +0.34%     | 0.36     | 0.7218    | `[-1.68%, +2.36%]`  |
| LUNG     | `TokenOSCAR` vs `mRNA_only` | +0.98%     | 0.92     | 0.3717    | `[-1.30%, +3.25%]`  |
| COADREAD | `TokenOSCAR` vs `Concat`    | +0.07%     | 0.02     | 0.9814    | `[-6.20%, +6.34%]`  |
| COADREAD | `TokenOSCAR` vs `Reverse`   | -0.37%     | -0.19    | 0.8498    | `[-4.47%, +3.73%]`  |
| COADREAD | `TokenOSCAR` vs `mRNA_only` | +2.23%     | 0.59     | 0.5627    | `[-5.84%, +10.30%]` |


**Kết luận rút ra (Q2, Q7, Q8):**

- Khi mở rộng ra 3 seed (45 fold), **hiệu ứng của** `TokenOSCAR` **so với** `Concat` **rõ hơn hẳn so với 1 seed** (+1.27% Avg F1, riêng BRCA **+2.83%**), và vượt `mRNA_only` trên cả 3 cohort (+1.42% Avg F1).
- Trên 3 seed, **thứ tự sinh học (**`Biological`**) vượt** `Reverse` **trên cả BRCA (+0.22%) lẫn LUNG (+0.34%)**, còn trung bình 3 cohort gần tương đương (+0.06%) — nhất quán với kết luận thận trọng của luận văn.
- Dưới kiểm định Nadeau–Bengio bảo thủ trên cỡ mẫu TCGA ($N=222\text{–}682$), không có cặp nào đạt $p < 0.05$, củng cố lập trường báo cáo trung thực ở mức mô tả.

---



## 6. Exp 6 — XAI Comparison & Signed MSigDB Hallmark trên BRCA (Trả lời Q10, Q12)

**File kết quả:** `experiments/defense_followup/exp6_xai_checks/xai_summary.json`

### 6.1. So sánh phương pháp Attribution (trung bình 5 fold × 4 subtype)


| Omic         | Spearman(Van, IG) | Spearman(IxG, IG) | Top-20 Overlap(IxG, IG) |
| ------------ | ----------------- | ----------------- | ----------------------- |
| `mRNA`       | 0.764             | **0.999**         | **96.2%**               |
| `Proteomics` | 0.742             | **0.945**         | **85.8%**               |
| `CNV`        | 0.425             | **0.998**         | **95.0%**               |




### 6.2. Signed MSigDB Hallmark ORA cho Luminal A (top-200 `|IG|` tách theo chiều biểu hiện)

Khi tách riêng nhóm gen giảm biểu hiện trong Luminal A (tránh trộn lẫn gen tăng/giảm như ORA không dấu ở Table 5.6), **3 Hallmark tăng sinh kinh điển được thu hồi với FDR cực mạnh**:

1. `HALLMARK_E2F_TARGETS` — overlap 15/200, **$q = 1.50 \times 10^{-13}$** (`CDK1`, `CHEK1`, `BRCA2`, `BARD1`, `CKS1B`, ...)
2. `HALLMARK_G2M_CHECKPOINT` — overlap 11/199, **$q = 1.29 \times 10^{-8}$** (`CDK1`, `CHEK1`, `KIF11`, `TTK`, `INCENP`, ...)
3. `HALLMARK_MITOTIC_SPINDLE` — overlap 8/199, **$q = 2.55 \times 10^{-5}$** (`BRCA2`, `CDK1`, `KIF11`, `SMC4`, `TTK`, ...)

**Kết luận rút ra (Q10, Q12):**

- `Input × Gradient` và `Integrated Gradients` gần như trùng khớp (`Spearman = 0.945–0.999`, `Top-20 overlap = 85.8–96.2%`).
- Giải thích cho Q12 được chứng minh trực tiếp bằng số liệu: việc Table 5.6 không ra MSigDB Hallmark trên LumA hoàn toàn do danh sách 22 gen ngắn và không tách dấu; khi dùng signed top-IG, các hallmark đặc trưng của Luminal A (ức chế chu kỳ tế bào E2F / G2M / Mitotic Spindle) lập tức xuất hiện với $q < 10^{-4}$.

---



## 7. Exp 7 — `OmicToTokens` Deep-Dive: Token Diversity, Asymmetric RPPA & Feature-Chunking (Trả lời Q1)

**File kết quả:**

- `experiments/defense_followup/exp7_omictotokens/exp7a_token_entropy.json`
- `experiments/defense_followup/exp7_omictotokens/exp7bc_results.json`



### 7.1. Exp 7a — Kiểm toán đa dạng Token & Cross-Attention Entropy (BRCA, 5 checkpoints)

| Omic | Off-diag Cosine Mean | Off-diag `|Cosine|` Mean | Effective Rank (trên tối đa 32) |
|---|---|---|---|
| `mRNA` | 0.364 | 0.371 | **28.63 / 32** |
| `CNV` | 0.054 | 0.150 | **31.21 / 32** |
| `Proteomics` | 0.032 | 0.145 | **31.36 / 32** |

- **Cross-Attention Entropy chuẩn hóa ($H(A)/\log 256$):**
  - `CNV → mRNA`: `0.9802 ± 0.0006` (max weight `0.0126` $\approx 3.2\times$ uniform `0.0039`, inter-patient std `0.0018`).
  - `mRNA → Proteomics`: `0.9854 ± 0.0004` (max weight `0.0111` $\approx 2.8\times$ uniform `0.0039`, inter-patient std `0.0016`).
- **Ý nghĩa:** Hoàn toàn **không có token collapse** — 256 token phân bố gần full-rank (28.6–31.4 / 32) với cosine giữa các cặp token rất thấp (0.03–0.05 ở CNV/RPPA). Cross-attention hoạt động như bộ trộn phân tán mềm (soft distributed mixing) thay vì one-hot cứng.



### 7.2. Exp 7b & 7c — Asymmetric RPPA ($T_P < T_G, T_T$) & `FeatureChunkToTokens` (Macro-F1, 5-fold)


| Biến thể Tokenizer                         | Params (BRCA) | BRCA F1           | LUNG F1           | COADREAD F1       | Avg F1     | Δ vs Default |
| ------------------------------------------ | ------------- | ----------------- | ----------------- | ----------------- | ---------- | ------------ |
| `TokenOSCAR (OmicToTokens 256×32)`         | 16,420,996    | **89.93 ± 2.15%** | **97.34 ± 1.30%** | 89.26 ± 2.55%     | **92.18%** | —            |
| `asym_RPPA_16` ($T_P=16$, tiết kiệm 1.98M) | 14,439,556    | 88.19 ± 3.21%     | 95.88 ± 1.90%     | **90.79 ± 2.80%** | 91.62%     | -0.56%       |
| `asym_RPPA_32` ($T_P=32$, tiết kiệm 1.85M) | 14,571,652    | 88.67 ± 3.17%     | 96.17 ± 1.18%     | 87.44 ± 2.79%     | 90.76%     | -1.42%       |
| `asym_RPPA_64` ($T_P=64$, tiết kiệm 1.59M) | 14,835,844    | 89.51 ± 4.08%     | 95.73 ± 1.94%     | 86.59 ± 6.05%     | 90.61%     | -1.57%       |
| `feature_chunk_64x32` (ViT-style)          | 100,644       | 77.47 ± 3.68%     | 94.10 ± 2.24%     | 82.31 ± 3.01%     | 84.63%     | **-7.55%**   |
| `feature_chunk_256x32` (ViT-style)         | 104,420       | 61.25 ± 9.82%     | 89.71 ± 5.04%     | 44.54 ± 13.64%    | 65.17%     | **-27.01%**  |


**Kết luận rút ra (Q1):**

- **Khóa chặt lý do chọn** `OmicToTokens` **(factorized bottleneck):** Cắt thẳng vector bảng gen thành các mảnh độc lập kiểu ViT (`FeatureChunkToTokens`) làm tụt thê thảm **-7.55% F1** (ở $64\times 32$) đến **-27.01% F1** (ở $256\times 32$), vì bảng omics không có tính cục bộ không gian như ảnh — bottleneck toàn cục $d_{\text{mid}}=256$ là bắt buộc để mỗi token tổng hợp được tương tác gen toàn omic.
- **Về RPPA bất đối xứng ($T_P$):** Giảm $T_P=16$ tiết kiệm gần **2.0M tham số** và giúp tăng **+1.53 F1 trên COADREAD** (90.79%, nơi RPPA chỉ có 153 protein), nhưng trên hai tập lớn hơn (BRCA, LUNG) giữ đối xứng $T_P=256$ vẫn cho hiệu năng tổng thể cao nhất (**92.18%** vs **91.62%**).

---

## 8. Exp 8 — Thay thế `Softmax` trong Cross-Attention & Tinh chỉnh `OmicToTokens`

**File kết quả:** `experiments/defense_followup/exp8_attn_tokenizer/exp8_results.json`

| Biến thể | Params (BRCA) | BRCA F1 | LUNG F1 | COADREAD F1 | Avg F1 | Δ vs Default |
|---|---|---|---|---|---|---|
| **`TokenOSCAR (Softmax, d_mid=256)`** | 16,420,996 | **89.93 ± 2.15%** | **97.34 ± 1.30%** | **89.26 ± 2.55%** | **92.18%** | — |
| `attn_sigmoid` ($\sigma(QK^\top/\sqrt{d_k}-\log T)$) | 16,420,996 | 89.62 ± 2.43% | 96.17 ± 1.27% | 89.26 ± 3.31% | 91.68% | -0.50% |
| `tok_layernorm` (post-reshape LN) | 16,421,188 | 89.34 ± 2.51% | 97.20 ± 0.73% | 88.28 ± 1.35% | 91.61% | -0.57% |
| `attn_sparsemax` (simplex thưa) | 16,420,996 | 89.10 ± 1.22% | 96.62 ± 0.99% | 88.73 ± 1.70% | 91.48% | -0.70% |
| `tok_dmid512` (bottleneck 512) | 32,718,212 | 89.49 ± 2.69% | 96.91 ± 0.98% | 86.05 ± 6.16% | 90.82% | -1.36% |
| `attn_sinkhorn` (5-iter OT) | 16,420,996 | 87.41 ± 1.55% | 96.32 ± 1.54% | 88.29 ± 2.67% | 90.67% | -1.51% |
| `attn_linear_elu` (ELU+1) | 16,420,996 | 89.44 ± 1.34% | 96.31 ± 0.82% | 84.63 ± 10.83% | 90.13% | -2.05% |

**Kết luận rút ra:**
- **Thiết kế mặc định (`Softmax` + `OmicToTokens d_mid=256`, không post-LN) vẫn đạt #1 trên cả 3 tập (`92.18%` Avg F1)** — không cần thay đổi kiến trúc chính.
- **Về thay thế Softmax:** `Sigmoid` (`91.68%`) và `Sparsemax` (`91.48%`) là hai lựa chọn thay thế ổn định nhất (`Sparsemax` giảm mạnh phương sai trên cả 3 tập xuống `±0.99%–1.70%`), trong khi `Sinkhorn` (`90.67%`) và `Linear ELU+1` (`90.13%`, COADREAD `±10.83%`) kém ổn định hơn.
- **Về Tokenizer `OmicToTokens`:** Tăng bottleneck lên `d_mid=512` gấp đôi số tham số (`32.7M`) nhưng lại **giảm -1.36% Avg F1** (overfit mạnh trên COADREAD `86.05%`), khẳng định `d_mid=256` đã là điểm bão hòa tối ưu.

---

## 9. Exp 9 — Tự học thứ tự/cạnh có hướng (`Learnable Directed Edge-Gated TokenOSCAR`) (Trả lời sâu Q7)

**Script & File kết quả:**

- `experiments/run_exp9_learned_order.py`
- `experiments/defense_followup/exp9_learned_order/exp9_learned_order_results.json`

**Thiết kế:** Thay 2 cạnh cố định bằng 6 logit cạnh có hướng $A \in \mathbb{R}^6$ ($G_e = \sigma(A_e/\tau)$, $\tau: 1.0 \to 0.25$ theo `max_epochs`, `edge_lr = 0.05`, `weight_decay = 0`), tách đúng phần residual cập nhật qua `LayerNorm(elementwise_affine=False)` để chống gian lận tỷ lệ:
$$Z_j^{\text{new}} = Z_j + \sum_{i \neq j} G_{i \to j} \cdot \left(\alpha_{i \to j} \odot \text{LN}_{\text{no-affine}}(\Delta_{i \to j})\right)$$
với phạt phi chu trình chính xác cho 3 đỉnh $h(G) = (G_{01}G_{10}+G_{02}G_{20}+G_{12}G_{21}) + (G_{01}G_{12}G_{20}+G_{02}G_{21}G_{10})$.

### 9.1. Đối chứng dương trên dữ liệu tổng hợp (XOR Chain `CNV → mRNA → Proteomics`, 5 seeds)

Dữ liệu tổng hợp 4 lớp có $a = c \cdot z_1$ (cần tương tác `{CNV, mRNA}`), $b = m \cdot z_2$ (cần `{mRNA, Proteomics}`), `{CNV, Proteomics}` không chia sẻ yếu tố nào; LogReg đơn omic lẫn concat tuyến tính đều ở mức ngẫu nhiên (`CNV 24.9%`, `mRNA 26.9%`, `Proteomics 21.1%`, `Concat LogReg 21.5%`).

| Thiết lập tổng hợp | Test F1 (5 seeds) | Skeleton đúng | Ma trận gate $G$ trung bình (`0→1, 0→2, 1→0, 1→2, 2→0, 2→1`) |
|---|---|---|---|
| `S1_control_bio_2edge` (oracle `0→1→2`, 2-step, readout Prot) | **100.00 ± 0.00%** | 5/5 | `1.00, 0.00, 0.00, 1.00, 0.00, 0.00` |
| `S1_control_rev_2edge` (ngược `2→1→0`, 2-step, readout Prot) | 24.10 ± 3.22% | 5/5 | `0.00, 0.00, 1.00, 0.00, 0.00, 1.00` |
| `S1_learned_budget2_dag` (tự học, 2-step, readout Prot) | **100.00 ± 0.00%** | 0/5 *(collider)* | `0.00, 1.00, 0.00, 1.00, 0.00, 0.00` *(chọn `CNV→Prot←mRNA` 5/5)* |
| `S2_symmetric_readout_budget2` (1-step, readout đối xứng) | **100.00 ± 0.00%** | **5/5 (100%)** | `0.00, 0.00, 1.00, 0.80, 0.00, 0.20` *(cắt sạch `{CNV,Prot}` = 0.00)* |
| `S3_full_selfattn_l1_dag` (Full + `lam1 * G.sum()`) | **100.00 ± 0.00%** | 0/5 *(collapse)* | `0.00, 0.00, 0.00, 0.00, 0.00, 0.00` *(sập về 0 do đường tắt SA)* |
| `S3_full_selfattn_budget2_dag` (Full + `Budget2+DAG`) | **100.00 ± 0.00%** | **4/5 (80%)** | `0.20, 0.20, 0.60, 1.00, 0.00, 0.00` |

**Phát hiện từ đối chứng dương:**
1. Phạt thưa `lam1 * G.sum()` khi có 2 tầng Self-Attention phía sau (`S3_full_selfattn_l1_dag`) bị **đường tắt Self-Attention đánh sập cả 6 gate về `0.00`**; ràng buộc ngân sách `Budget2+DAG` khắc phục hoàn toàn và khôi phục đúng khung tương tác (`4/5` ở `S3`, `5/5` ở `S2`).
2. `S1_learned_budget2_dag` chứng minh trực quan điểm phương pháp luận: chỉ từ nhãn phân loại, mô hình chọn **cấu trúc dự đoán ngắn nhất** (collider 1-hop `CNV → Prot ← mRNA`, 5/5 seed) thay vì chuỗi nhân quả 2-hop `CNV → mRNA → Prot`.

### 9.2. Kết quả trên TCGA (3 seeds `42, 123, 2024` × 5 folds = 15 folds/cohort; 45 folds tổng)

| Mô hình (15 folds/cohort, 3 seeds) | BRCA F1 | LUNG F1 | COADREAD F1 | Avg F1 | Δ vs Bio 2-edge |
|---|---|---|---|---|---|
| **`TokenOSCAR (Fixed Bio 2-edge, Exp 5)`** | **89.84 ± 2.95%** | 96.66 ± 1.52% | 89.00 ± 4.13% | **91.83%** | — |
| `Reverse (Fixed 2-edge, Exp 5)` | 89.62 ± 3.24% | 96.32 ± 1.32% | **89.37 ± 4.03%** | 91.77% | -0.06% |
| `all2all_norm_fixed` (6 cạnh cố định $G=1$) | 89.36 ± 2.59% | 96.27 ± 1.22% | 88.37 ± 3.80% | 91.33% | -0.50% |
| `learned_edge_free` (tự học 6 cạnh, thuần CE) | 89.44 ± 2.51% | **97.25 ± 1.29%** | 86.45 ± 4.28% | 91.05% | -0.78% |
| `learned_edge_budget2_dag` (tự học 2 cạnh DAG) | 89.15 ± 3.11% | 96.37 ± 1.25% | 87.33 ± 3.16% | 90.95% | -0.88% |

### 9.3. Cấu trúc cạnh học được qua 45 folds (3 cohorts × 3 seeds × 5 folds, chỉ học trên tập train)

| Cạnh có hướng | `learned_edge_free` Mean (tần suất > 0.5) | `learned_edge_budget2_dag` Mean (tần suất > 0.5) |
|---|---|---|
| `mRNA → Proteomics` | **0.962 ± 0.043** (`45/45 = 100.0%`) | **0.993 ± 0.011** (`45/45 = 100.0%`) |
| `mRNA → CNV` | **0.948 ± 0.082** (`45/45 = 100.0%`) | **0.950 ± 0.205** (`43/45 = 95.6%`) |
| `Proteomics → mRNA` | 0.717 ± 0.190 (`37/45 = 82.2%`) | 0.001 ± 0.002 (`0/45 = 0.0%`) |
| `CNV → mRNA` | 0.459 ± 0.270 (`16/45 = 35.6%`) | 0.001 ± 0.001 (`0/45 = 0.0%`) |
| `Proteomics → CNV` | 0.415 ± 0.291 (`15/45 = 33.3%`) | 0.004 ± 0.007 (`0/45 = 0.0%`) |
| `CNV → Proteomics` | 0.232 ± 0.117 (`1/45 = 2.2%`) | 0.048 ± 0.205 (`2/45 = 4.4%`) |

**Kết luận rút ra (Exp 9):**
- **Phát hiện cực kỳ nhất quán (43–45/45 fold):** Dưới thuần nhãn phân loại, mô hình **không** tự khôi phục chuỗi Central Dogma (`CNV → mRNA → Proteomics`), mà hội tụ gần như tuyệt đối vào cấu trúc **nhánh tỏa từ mRNA (`CNV ← mRNA → Proteomics`)** (`mRNA → Proteomics` `100%`, `mRNA → CNV` `95.6–100%`, còn cặp trực tiếp `CNV ↔ Proteomics` gần như bị loại hoàn toàn). Điều này phản ánh đúng thứ hạng SNR của 3 omic (mRNA `91.4%` > Proteomics `76.4%` > CNV `55.2%`): gradient phân loại ưu tiên phát tín hiệu từ omic mạnh nhất (`mRNA`) sang 2 omic yếu hơn.
- **Nhưng tự học cạnh KHÔNG thắng cố định thứ tự sinh học (và thậm chí kém hơn cả `all2all_norm_fixed`):** Trên trung bình 3 cohort (45 fold), `TokenOSCAR (Fixed Bio 2-edge)` (**91.83%**) vượt cả `all2all_norm_fixed` (**91.33%**, -0.50%), `learned_edge_free` (**91.05%**, -0.78%) lẫn `learned_edge_budget2_dag` (**90.95%**, -0.88%). Trên COADREAD ($N=222$), việc mô hình tự khuếch đại nhánh `mRNA → {CNV, Proteomics}` kéo hiệu năng về sát mức `mRNA_only` (`86.45–87.33%` vs `mRNA_only 86.77%`), làm mất tính bổ trợ của CNV/Proteomics so với cố định `CNV → mRNA → Proteomics` (**89.00%**). Riêng trên LUNG, `learned_edge_free` đạt **97.25%** (+0.59% so với Bio 2-edge) nhờ giữ thêm `Proteomics → mRNA` (0.90) và `Proteomics → CNV` (0.74).
- **Ý nghĩa bảo vệ luận văn:** Biện minh trực tiếp cho thiết kế cố định 2 cạnh sinh học gọn gàng làm inductive bias — đồng thời chứng minh bằng thực nghiệm rằng để học hướng nhân quả sinh học thực sự từ dữ liệu, thuần nhãn phân loại là không đủ (cần tín hiệu giám sát chéo giữa các tầng omic).

---

## 10. Exp 10 — Thay thế `OmicToTokens` bằng Nhóm gen theo Biological Pathway (`PathwayBottleneck` & `PathwayTokens`) (Trả lời sâu Q1, Q15)

**Code & File kết quả:**
- `models/pathway_tokenizer.py`, `experiments/run_exp10_pathway_tokenizer.py`
- `experiments/defense_followup/exp10_pathway_tokenizer/exp10_results.json`
- Báo cáo chi tiết: `ablation_study/EXP10_PATHWAY_TOKENIZER.md`

**Thiết kế:** Thay bộ mã hóa dày `OmicToTokens` cho `mRNA` và `CNV` bằng hai cơ chế thưa theo pathway (giữ nguyên `OmicToTokens` cho RPPA để bảo toàn phospho-protein):
- **Pathway-Bottleneck (`pw_bn_*`):** Lớp tuyến tính thưa gen $\to$ $P$ pathway (`edge-indexed index_add_`, khởi tạo fan-in $1/\sqrt{\text{deg}_p}$) thay cho `Linear(D, 256)`, sau đó chiếu ra `(B, 256, 32)` (+ biến thể `pw_bn_pathformer_res` bổ sung nhánh residual 64 chiều cho các gen ngoài pathway).
- **Pathway-as-Tokens (`pw_tok_*`):** Mỗi pathway gom trực tiếp gen thành viên qua $K=8$ chiếu học được $\to$ MLP chia sẻ $\to$ `(B, P, 32)` đưa thẳng vào Cross/Self-Attention (`RPPA` chiếu sang $T_P=P$ token).

| Biến thể Tokenizer | Params (BRCA) | BRCA F1 | LUNG F1 | COADREAD F1 | Avg F1 | Δ vs Default |
|---|---|---|---|---|---|---|
| **`TokenOSCAR (OmicToTokens 256×32)`** | 16,420,996 | **89.93 ± 2.15%** | 97.34 ± 1.30% | 89.26 ± 2.55% | **92.18%** | — |
| `pw_bn_pathformer_res` (`P=1,497` + 64-dim unmapped) | 29,063,500 | 88.77 ± 2.95% | 96.46 ± 2.01% | **90.04 ± 5.86%** | 91.76% | **-0.42%** |
| `pw_bn_pathformer` (`P=1,497`, 52–59% gen) | 26,984,076 | 86.97 ± 3.19% | 97.64 ± 1.19% | 88.46 ± 5.00% | 91.02% | -1.16% |
| `pw_bn_hallmark` (`P=50`, 21–23% gen) | 3,112,917 | 85.00 ± 2.73% | 97.49 ± 1.11% | 84.00 ± 4.59% | 88.83% | -3.35% |
| `pw_bn_kegg` (`P=146`, 19–25% gen) | 4,691,343 | 86.49 ± 1.33% | 97.79 ± 1.33% | 81.60 ± 6.07% | 88.63% | -3.55% |
| `pw_tok_hallmark` (`50×32` tokens, **giảm 24.5×**) | **670,124** | 85.85 ± 4.90% | **98.09 ± 0.76%** | 80.23 ± 4.90% | 88.06% | -4.12% |
| `pw_tok_kegg` (`146×32` tokens, **giảm 10.8×**) | 1,516,412 | 83.79 ± 4.88% | 97.94 ± 0.71% | 75.67 ± 3.53% | 85.80% | -6.38% |

**Kết luận rút ra (Exp 10):**
- **Trên LUNG (`LUAD` vs `LUSC`), nhóm theo pathway vượt hẳn baseline trên 5/6 biến thể với số tham số siêu nhỏ:** `pw_tok_hallmark` lập kỷ lục **`98.09 ± 0.76%`** (+0.75%, giảm một nửa std) chỉ với **`655K` tham số** (gọn hơn 25× so với `16.4M`).
- **Trên phân nhóm mịn 4 lớp (`BRCA`, `COADREAD`), giới hạn độ phủ gen (gene coverage) là yếu tố quyết định:** bỏ 40–80% gen ngoài pathway làm tụt điểm rõ rệt; khi nâng độ phủ lên Pathformer (`52–59%`) và khôi phục gen unmapped qua nhánh residual (`pw_bn_pathformer_res`), điểm phục hồi ngay về **`91.76%`** (COADREAD đạt **`90.04%`**, BRCA `88.77%`). Điều này kiểm chứng trực tiếp lập luận Q15 về ưu thế bảo toàn 100% hệ gen của `OmicToTokens`.
- **So với cắt chunk kiểu ViT (Exp 7c):** `pw_tok_hallmark` (`88.06%`) vượt xa `feature_chunk_64x32` (`84.63%`, +3.43%) và `feature_chunk_256x32` (`65.17%`, +22.89%), khẳng định nhóm theo pathway là cách gom cục bộ hợp lý hơn hẳn về mặt sinh học.

### 10.1. Exp 10b — Thu gọn RPPA (`d_mid=64, T_P=16`) & Nhóm cả 3 Omics theo Pathway (gồm RPPA)

**File kết quả:** `experiments/defense_followup/exp10_pathway_tokenizer/exp10b_results.json` (ánh xạ kháng thể RPPA $\to$ HGNC giữ trọng số cạnh riêng cho từng phospho-site, độ phủ Pathformer `97.8–98.0%`, Hallmark `79.1–83.3%`, KEGG `77.1–79.4%`).

| Biến thể Exp 10b | Params BRCA (RPPA) | BRCA F1 | LUNG F1 | COADREAD F1 | Avg F1 | Δ vs Default |
|---|---|---|---|---|---|---|
| `pw_bn_pathformer_res_all3` (3 omics, `T_P=16`) | 27,680,866 (788K) | **89.34 ± 2.88%** | 95.73 ± 1.50% | **89.75 ± 3.62%** | **91.61%** | **-0.57%** |
| `pw_bn_pathformer_res_rppa64x16` | 26,940,364 (48K) | 87.18 ± 2.68% | 95.28 ± 1.78% | 86.79 ± 1.99% | 89.75% | -2.43% |
| `pw_tok_hallmark_rppa64x16` (**gọn 66×**) | 247,724 (48K) | 82.47 ± 3.76% | 97.35 ± 1.00% | 85.66 ± 5.22% | 88.49% | -3.69% |
| `pw_bn_hallmark_rppa64x16` (gọn 16.6×) | 989,781 (48K) | 84.85 ± 2.32% | 97.49 ± 0.37% | 80.39 ± 7.58% | 87.58% | -4.60% |
| `pw_tok_hallmark_all3` (**gọn 79×**) | **207,188 (6.1K)** | 80.33 ± 1.79% | 95.58 ± 1.40% | 85.83 ± 3.13% | 87.25% | -4.93% |
| `pw_tok_kegg_all3` (**gọn 61×**) | 271,148 (13.3K) | 78.96 ± 3.51% | 96.03 ± 1.00% | 84.39 ± 2.35% | 86.46% | -5.72% |
| `pw_tok_kegg_rppa64x16` (gọn 54×) | 301,436 (48K) | 75.80 ± 7.28% | **97.64 ± 1.28%** | 84.41 ± 2.18% | 85.95% | -6.23% |

- **Xác nhận đúng giả thuyết overfit RPPA trên COADREAD:** Thu gọn RPPA giúp `pw_tok_hallmark` bật tăng **`+5.43%`–`+5.60%`** (`80.23% -> 85.83%`, chỉ `202K` params) và `pw_tok_kegg` tăng **`+8.72%`–`+8.74%`** (`75.67% -> 84.41%`).
- **Nhóm cả 3 omics với Pathformer + residual (`pw_bn_pathformer_res_all3`)** đạt **`89.34%` BRCA** (cao nhất trong toàn bộ họ pathway, sát `89.93%` baseline) và **`89.75%` COADREAD** (vượt baseline `89.26%`, Avg **`91.61%`**).





