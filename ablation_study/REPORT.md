# TokenOSCAR (`256×32`) — Ablation Study Report (Table 5.2 & Table 5.3)

## 1. Mục tiêu & Thiết lập

Báo cáo này tái hiện các thí nghiệm ablation trong Chương 5 của luận văn (`main.pdf`, **Table 5.2** và **Table 5.3**) trên kiến trúc **TokenOSCAR (`T = 256, M = 32`, `d_mid = 256`, `n_heads = 4`, 2-layer Transformer)**, bổ sung thêm hai biến thể thành phần (`Without cross-attention` và `Without gate & self-attention`).

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
- **Concatenation:** bỏ cả ordered cross-attention, gated fusion và Transformer self-attention; mean-pool từng omic `(B, 256, 32) → (B, 32)`, nối 3 omic `(B, 96)` rồi đưa vào MLP classifier.
- **Without cross-attention:** bỏ ordered cross-attention và gated fusion; giữ nguyên 2-layer Transformer self-attention trên chuỗi `769` token (`[CLS] + 3×256`).
- **Without self-attention:** giữ ordered cross-attention và gated fusion; bỏ 2-layer Transformer self-attention, thay bằng mean-pool từng omic rồi nối `(B, 96)` đưa vào MLP classifier.
- **Without gated fusion:** giữ ordered cross-attention và Transformer self-attention; bỏ `TokenGatedFusion`, đẩy thẳng đầu ra cross-attention đi tiếp (`F = H_enh`).
- **Without gate & self-attention:** giữ ordered cross-attention; bỏ cả `TokenGatedFusion` lẫn Transformer self-attention (chỉ còn cross-attention → mean-pool concat → classifier).
- **TokenOSCAR (full):** bản đầy đủ `CNV → mRNA → Proteomics` với ordered cross-attention, token-wise gated fusion và 2-layer Transformer self-attention.

### Accuracy (%, mean ± std)

| Model variant | BRCA | LUNG | COADREAD | Avg Acc |
|---|---:|---:|---:|---:|
| Concatenation | 90.02 ± 1.30 | 95.90 ± 1.43 | 90.08 ± 3.70 | 92.00 |
| Without cross-attention | 91.00 ± 1.69 | **96.77 ± 1.51** | 91.45 ± 4.31 | **93.08** |
| Without self-attention | 90.22 ± 1.64 | 95.02 ± 1.49 | 87.84 ± 3.04 | 91.02 |
| Without gated fusion | **91.19 ± 2.25** | 95.75 ± 1.94 | 89.66 ± 2.98 | 92.20 |
| Without gate & self-attention | 90.80 ± 1.35 | 95.90 ± 1.77 | 85.17 ± 4.50 | 90.62 |
| **TokenOSCAR (full)** | 91.00 ± 1.69 | 94.73 ± 3.73 | **92.33 ± 2.73** | 92.69 |

### Macro-F1 (%, mean ± std)

| Model variant | BRCA | LUNG | COADREAD | Avg F1 |
|---|---:|---:|---:|---:|
| Concatenation | 88.85 ± 2.08 | 95.87 ± 1.44 | 89.61 ± 4.79 | 91.44 |
| Without cross-attention | 89.56 ± 2.27 | **96.76 ± 1.52** | 88.81 ± 6.31 | 91.71 |
| Without self-attention | 88.71 ± 1.83 | 94.99 ± 1.50 | 86.72 ± 4.12 | 90.14 |
| Without gated fusion | 89.59 ± 2.75 | 95.73 ± 1.94 | 87.11 ± 3.94 | 90.81 |
| Without gate & self-attention | **89.90 ± 2.58** | 95.87 ± 1.79 | 83.04 ± 5.79 | 89.60 |
| **TokenOSCAR (full)** | 89.31 ± 2.17 | 94.70 ± 3.76 | **91.34 ± 2.55** | **91.78** |

---

## 3. Table 5.3 — Effect of Cross-Attention Order

### Accuracy (%, mean ± std)

| Ordering | BRCA | LUNG | COADREAD | Avg Acc |
|---|---:|---:|---:|---:|
| Proteomics → transcriptomics → genomics (reverse) | **91.58 ± 1.83** | **96.48 ± 1.56** | 90.99 ± 2.01 | **93.02** |
| Transcriptomics → genomics → proteomics (mixed) | **91.58 ± 2.12** | 96.34 ± 1.14 | 89.65 ± 3.06 | 92.52 |
| Genomics → transcriptomics → proteomics (TokenOSCAR) | 91.00 ± 1.69 | 94.73 ± 3.73 | **92.33 ± 2.73** | 92.69 |

### Macro-F1 (%, mean ± std)

| Ordering | BRCA | LUNG | COADREAD | Avg F1 |
|---|---:|---:|---:|---:|
| Proteomics → transcriptomics → genomics (reverse) | **90.40 ± 2.37** | **96.46 ± 1.57** | 89.35 ± 2.35 | **92.07** |
| Transcriptomics → genomics → proteomics (mixed) | 90.27 ± 2.88 | 96.31 ± 1.15 | 86.70 ± 4.73 | 91.09 |
| Genomics → transcriptomics → proteomics (TokenOSCAR) | 89.31 ± 2.17 | 94.70 ± 3.76 | **91.34 ± 2.55** | 91.78 |

---

## 4. Nhận xét chính

1. **Vai trò của Self-Attention:** Bỏ self-attention (`Without self-attention` và `Without gate & self-attention`) làm giảm mạnh nhất trên trung bình cả 3 cohort (Avg F1 tụt `-1.64` và `-2.18`), đặc biệt nặng trên COADREAD (`-4.62` và `-8.30` F1).
2. **Vai trò của Gated Fusion:** Trên COADREAD (N = 222, 4 lớp CMS), bỏ gate làm tụt `-4.23` F1 / `-2.67` Acc và tăng độ dao động (std F1 từ `2.55` lên `3.94`). Trên BRCA, bỏ gate gần như hòa về điểm trung bình (`+0.28` F1 / `+0.19` Acc) nhưng kém ổn định hơn (std F1 `2.75` vs `2.17`), do `TokenCrossAttention` đã có sẵn residual connection.
3. **Vai trò của Cross-Attention:** Biến thể `Without cross-attention` (chỉ dùng Transformer self-attention) đạt Avg Acc cao nhất (**93.08%**, đặc biệt mạnh trên LUNG `96.77%`), nhưng trên COADREAD lại giảm F1 (`88.81%` vs `91.34%`) và dao động rất lớn (`std 6.31` vs `2.55`). Nhờ đó `TokenOSCAR (full)` vẫn giữ Avg F1 cao nhất trong nhóm component ablation (**91.78%**).
4. **Ảnh hưởng của thứ tự (Table 5.3):** Khác với OSCAR `1×128` trong `main.pdf`, trên TokenOSCAR `256×32`, thứ tự sinh học `CNV → mRNA → Proteomics` chỉ thắng rõ trên **COADREAD** (`92.33%` Acc / `91.34%` F1). Trên **BRCA**, cả `reverse` và `mixed` đều nhỉnh hơn nhẹ (`91.58%` vs `91.00%` Acc). Trên **LUNG**, `TokenOSCAR (full)` bị kéo xuống do fold 1 dừng sớm ở `chosen_epoch = 1` (`87.59%` Acc).

---

## 5. Lệnh tái lập

```bash
# Chạy toàn bộ 8 biến thể ablation (Table 5.2 mở rộng & Table 5.3) trên 3 cohort
uv run python train_token_oscar_ablation.py --cohort all --variants all
```

Dữ liệu chi tiết từng fold được lưu tại `ablation_study/results.json`.
