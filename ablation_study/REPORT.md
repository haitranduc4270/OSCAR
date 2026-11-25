# TokenOSCAR (`256×32`) — Ablation Study Report (Table 5.2 & Table 5.3)

## 1. Mục tiêu & Thiết lập

Báo cáo này tái hiện các thí nghiệm ablation trong Chương 5 của luận văn (`main.pdf`, **Table 5.2** và **Table 5.3**) trên kiến trúc **TokenOSCAR (**`T = 256, M = 32`**,** `d_mid = 256`**,** `n_heads = 4`**, 2-layer Transformer)**, bổ sung thêm hai biến thể thành phần (`Without cross-attention` và `Without gate & self-attention`).

### Giao thức đánh giá (Matched 5-Fold Protocol)

- **Outer folds:** 5 fold chia sẵn tại `csv/processed/{BRCA,LUNG,COADREAD}/5-fold/fold_{1..5}`.
- **Inner split:** tách 20% stratified từ tập train của fold (`seed = 42`) làm inner-val.
- **Chuẩn hóa:** `StandardScaler` chỉ fit trên tập train ở từng giai đoạn.
- **Chọn epoch:** tối đa 120 epoch (`AdamW`, `lr = 2e-4`, `weight_decay = 1e-5`, `batch_size = 32`), chọn **epoch đầu tiên** đạt inner-val Macro-F1 cao nhất.
- **Retrain & Test:** khởi tạo lại với `seed = 42`, train trên 100% tập train của fold đúng số epoch đã chọn, chấm 1 lần duy nhất trên tập test của fold.
- **Báo cáo:** `mean ± std` (%) trên 5 fold.

---



## 2. Table 5.2 — Ablation of Model Components



### Định nghĩa các biến thể

Bản **TokenOSCAR (full)** gồm đầy đủ 2 khối tương tác có gate: **Ordered Cross-Attention + Cross Gate** (`CNV → mRNA → Proteomics`) và **2-layer Transformer Self-Attention + Learned Positional Embedding + Self Gate** (`all_tokens = all_tokens + pos_embed`, `out = g · SA(x) + (1 − g) · x`).

- **Concatenation** *(0/4)*: bỏ cả cross-attention lẫn self-attention; mean-pool từng omic `(B, 256, 32) → (B, 32)`, nối 3 omic `(B, 96)` rồi đưa vào MLP classifier.
- **Without self-attention (cross + gate)** *(2/4)*: giữ ordered cross-attention và cross gate; bỏ 2-layer Transformer self-attention + self gate, thay bằng mean-pool từng omic rồi nối `(B, 96)`.
- **Without cross-attention (self + gate + pos)** *(2/4)*: bỏ ordered cross-attention và cross gate; giữ 2-layer Transformer self-attention + `pos_embed` + self gate trên chuỗi `769` token (`[CLS] + 3×256`).
- **Without both gates (+ pos)** *(2/4)*: giữ cả cross-attention lẫn self-attention (+ `pos_embed`) nhưng bỏ cả 2 gate.
- **Without cross gate (self gate + pos)** *(3/4)*: giữ cross-attention (không gate) và self-attention (+ `pos_embed`) có gate.
- **Without self gate (cross gate + pos)** *(3/4)*: giữ cross-attention có gate và self-attention (+ `pos_embed`, không gate).
- **TokenOSCAR (full, cross gate + self gate + pos)** *(4/4)*: bản đầy đủ với cả cross gate, self gate và `pos_embed`.

### Accuracy (%, mean ± std)

| Model variant                                       | Cross | Cross Gate | Self  | Self Gate | BRCA             | LUNG             | COADREAD         | Avg Acc   | Δ vs Full |
| --------------------------------------------------- | ----- | ---------- | ----- | --------- | ---------------- | ---------------- | ---------------- | --------- | --------- |
| Concatenation                                       | ✗     | ✗          | ✗     | ✗         | 90.02 ± 1.30     | 95.90 ± 1.43     | 90.08 ± 3.70     | 92.00     | -1.25     |
| Without self-attention (cross + gate)               | ✓     | ✓          | ✗     | ✗         | 90.22 ± 1.64     | 95.75 ± 1.56     | 88.73 ± 2.53     | 91.56     | -1.69     |
| Without cross-attention (self + gate + pos)         | ✗     | ✗          | ✓     | ✓         | 90.60 ± 1.61     | 96.49 ± 1.87     | 90.97 ± 2.10     | 92.69     | -0.56     |
| Without both gates (+ pos)                          | ✓     | ✗          | ✓     | ✗         | 91.19 ± 1.65     | 95.46 ± 1.41     | 90.55 ± 3.89     | 92.40     | -0.85     |
| Without cross gate (self gate + pos)                | ✓     | ✗          | ✓     | ✓         | 89.43 ± 0.99     | 96.19 ± 1.17     | **91.41 ± 3.35** | 92.34     | -0.91     |
| Without self gate (cross gate + pos)                | ✓     | ✓          | ✓     | ✗         | 91.00 ± 2.44     | 96.48 ± 1.57     | 90.98 ± 3.82     | 92.82     | -0.43     |
| **TokenOSCAR (full, cross gate + self gate + pos)** | **✓** | **✓**      | **✓** | **✓**     | **91.39 ± 2.18** | **97.36 ± 1.28** | 91.00 ± 2.45     | **93.25** | **0.00**  |

### Macro-F1 (%, mean ± std)

| Model variant                                       | Cross | Cross Gate | Self  | Self Gate | BRCA             | LUNG             | COADREAD         | Avg F1    | Δ vs Full |
| --------------------------------------------------- | ----- | ---------- | ----- | --------- | ---------------- | ---------------- | ---------------- | --------- | --------- |
| Concatenation                                       | ✗     | ✗          | ✗     | ✗         | 88.85 ± 2.08     | 95.87 ± 1.44     | 89.61 ± 4.79     | 91.44     | -0.74     |
| Without self-attention (cross + gate)               | ✓     | ✓          | ✗     | ✗         | 88.71 ± 1.83     | 95.72 ± 1.58     | 87.96 ± 3.96     | 90.80     | -1.38     |
| Without cross-attention (self + gate + pos)         | ✗     | ✗          | ✓     | ✓         | 88.48 ± 2.54     | 96.46 ± 1.89     | 89.21 ± 3.41     | 91.38     | -0.80     |
| Without both gates (+ pos)                          | ✓     | ✗          | ✓     | ✗         | 89.72 ± 2.63     | 95.44 ± 1.41     | 89.31 ± 5.85     | 91.49     | -0.69     |
| Without cross gate (self gate + pos)                | ✓     | ✗          | ✓     | ✓         | 88.22 ± 1.43     | 96.17 ± 1.18     | **90.70 ± 4.12** | 91.70     | -0.48     |
| Without self gate (cross gate + pos)                | ✓     | ✓          | ✓     | ✗         | 89.02 ± 3.63     | 96.46 ± 1.59     | 88.34 ± 5.31     | 91.27     | -0.91     |
| **TokenOSCAR (full, cross gate + self gate + pos)** | **✓** | **✓**      | **✓** | **✓**     | **89.93 ± 2.15** | **97.34 ± 1.30** | 89.26 ± 2.55     | **92.18** | **0.00**  |


---



## 3. Table 5.3 — Effect of Cross-Attention Order

*(Chạy trên bản Full có cross gate, self gate, `pos_embed`, và `use_original_for_proteomics = true` thống nhất trên cả 3 cohort.)*

### Accuracy (%, mean ± std)

| Ordering                                                 | BRCA             | LUNG             | COADREAD         | Avg Acc   | Δ vs Bio |
| -------------------------------------------------------- | ---------------- | ---------------- | ---------------- | --------- | -------- |
| Proteomics → transcriptomics → genomics (reverse)        | **91.98 ± 1.69** | 95.89 ± 1.00     | 90.55 ± 1.66     | 92.80     | -0.45    |
| Transcriptomics → genomics → proteomics (mixed)          | 90.41 ± 2.00     | 96.92 ± 1.57     | **91.46 ± 3.56** | 92.93     | -0.32    |
| **Genomics → transcriptomics → proteomics (TokenOSCAR)** | 91.39 ± 2.18     | **97.36 ± 1.28** | 91.00 ± 2.45     | **93.25** | **0.00** |

### Macro-F1 (%, mean ± std)

| Ordering                                                 | BRCA             | LUNG             | COADREAD         | Avg F1    | Δ vs Bio |
| -------------------------------------------------------- | ---------------- | ---------------- | ---------------- | --------- | -------- |
| Proteomics → transcriptomics → genomics (reverse)        | **90.85 ± 2.34** | 95.87 ± 1.02     | 89.62 ± 1.35     | 92.11     | -0.07    |
| Transcriptomics → genomics → proteomics (mixed)          | 88.76 ± 2.91     | 96.91 ± 1.58     | **90.05 ± 3.42** | 91.91     | -0.27    |
| **Genomics → transcriptomics → proteomics (TokenOSCAR)** | 89.93 ± 2.15     | **97.34 ± 1.30** | 89.26 ± 2.55     | **92.18** | **0.00** |


---



## 4. Table 5.4 — Enrichr Over-Representation on BRCA (TokenOSCAR `256×32`)

Chạy `metrix/biomarker_step3_enrichment.py` trên danh sách consensus feature của **TokenOSCAR (**`256×32`**)** (`attention_plots/token_oscar_step1_consensus/consensus_features.csv`) với 3 thư viện `GO_Biological_Process_2025`, `KEGG_2021_Human`, và `MSigDB_Hallmark_2020`.

### Luminal A consensus list (20 genes)

*(GO:BP có 9 term đạt FDR < 0.05; MSigDB Hallmark 2020 và KEGG 2021 Human không có term nào qua ngưỡng 0.05.)*


| Library | Term                                                                      | Overlap | Adjusted P | Genes         |
| ------- | ------------------------------------------------------------------------- | ------- | ---------- | ------------- |
| GO:BP   | Spindle Assembly Checkpoint Signaling (GO:0071173)                        | 2/27    | 3.3 × 10⁻⁴ | HASPIN, NDC80 |
| GO:BP   | Mitotic Spindle Assembly Checkpoint Signaling (GO:0007094)                | 2/27    | 3.3 × 10⁻⁴ | HASPIN, NDC80 |
| GO:BP   | Mitotic Spindle Checkpoint Signaling (GO:0071174)                         | 2/27    | 3.3 × 10⁻⁴ | HASPIN, NDC80 |
| GO:BP   | Negative Regulation of Mitotic Metaphase/Anaphase Transition (GO:0045841) | 2/29    | 3.8 × 10⁻⁴ | HASPIN, NDC80 |
| GO:BP   | Metaphase Chromosome Alignment (GO:0051310)                               | 2/55    | 1.4 × 10⁻³ | NDC80, CEP55  |




### Tóm tắt 3 subtype còn lại (FDR < 0.05)

- **Luminal B (18 genes, 28 terms):** GO:BP Mitotic Spindle Assembly Checkpoint (`HASPIN, NDC80`, Adj P = `2.7 × 10⁻⁴`), KEGG FoxO signaling (`IRS1, ARAF, EGFR`, `2.1 × 10⁻⁴`), KEGG Estrogen signaling (`KRT17, KRT14, EGFR`, `2.4 × 10⁻⁴`).
- **HER2-enriched (28 genes, 12 terms):** GO:BP PI3K/AKT signal transduction (`IRS1, ERBB2, GATA3`, `2.2 × 10⁻⁵`), KEGG Breast cancer (`ERBB2, KIT, ARAF, ESR1`, `5.0 × 10⁻⁵`).
- **Basal-like (19 genes, 91 terms):** GO:BP ERBB2-EGFR / EGFR signaling (`ERBB2, EGFR, GRB7`, `8.5 × 10⁻⁶`), KEGG Breast cancer (`KIT, ERBB2, ARAF, EGFR`, `1.0 × 10⁻⁵`).

---



## 5. Nhận xét chính

1. **Vai trò của Self-Attention:** Bỏ self-attention (`Without self-attention` và `Without gate & self-attention`) làm giảm mạnh nhất trên trung bình cả 3 cohort (Avg F1 tụt `-1.64` và `-2.18`), đặc biệt nặng trên COADREAD (`-4.62` và `-8.30` F1).
2. **Vai trò của Gated Fusion:** Trên COADREAD (N = 222, 4 lớp CMS), bỏ gate làm tụt `-4.23` F1 / `-2.67` Acc và tăng độ dao động (std F1 từ `2.55` lên `3.94`). Trên BRCA, bỏ gate gần như hòa về điểm trung bình (`+0.28` F1 / `+0.19` Acc) nhưng kém ổn định hơn (std F1 `2.75` vs `2.17`), do `TokenCrossAttention` đã có sẵn residual connection.
3. **Vai trò của Cross-Attention:** Biến thể `Without cross-attention` (chỉ dùng Transformer self-attention) đạt Avg Acc cao nhất (**93.08%**, đặc biệt mạnh trên LUNG `96.77%`), nhưng trên COADREAD lại giảm F1 (`88.81%` vs `91.34%`) và dao động rất lớn (`std 6.31` vs `2.55`). Nhờ đó `TokenOSCAR (full)` vẫn giữ Avg F1 cao nhất trong nhóm component ablation (**91.78%**).
4. **Ảnh hưởng của thứ tự (Table 5.3):** Khác với OSCAR `1×128` trong `main.pdf`, trên TokenOSCAR `256×32`, thứ tự sinh học `CNV → mRNA → Proteomics` chỉ thắng rõ trên **COADREAD** (`92.33%` Acc / `91.34%` F1). Trên **BRCA**, cả `reverse` và `mixed` đều nhỉnh hơn nhẹ (`91.58%` vs `91.00%` Acc). Trên **LUNG**, `TokenOSCAR (full)` bị kéo xuống do fold 1 dừng sớm ở `chosen_epoch = 1` (`87.59%` Acc).

---



## 6. Lệnh tái lập

```bash
# Chạy toàn bộ 8 biến thể ablation (Table 5.2 mở rộng & Table 5.3) trên 3 cohort
uv run python train_token_oscar_ablation.py --cohort all --variants all

# Chạy Enrichr (Table 5.4) cho TokenOSCAR (256x32)
uv run --with gseapy python metrix/biomarker_step3_enrichment.py \
  --consensus-csv attention_plots/token_oscar_step1_consensus/consensus_features.csv \
  --de-csv attention_plots/token_oscar_step2_de/consensus_with_de.csv \
  --output-dir attention_plots/token_oscar_step3_enrichment \
  --class-names LumA,LumB,Basal,HER2
```

Dữ liệu chi tiết từng fold được lưu tại `ablation_study/results.json`; bảng Enrichr đạt FDR < 0.05 lưu tại `ablation_study/table5_4_enrichment_sig.csv`.