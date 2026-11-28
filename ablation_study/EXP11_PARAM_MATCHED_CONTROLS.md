# Exp 11 — Control Khớp Số Tham Số (~16.4M BRCA / ~18.1M LUNG & COADREAD)

## 1. Mục tiêu

Tách rạch ròi lợi ích đến từ **dung lượng tham số** khỏi lợi ích thực sự của **biểu diễn đa token (`256×32`) + Cross/Self-Attention có gate** trong `TokenOSCAR`.

Cụ thể, chúng ta mở rộng độ rộng nút cổ chai `d_mid` của các mô hình đối chứng (`Concat no-attention`, `mRNA_only_mlp`, `1×128 Full`) sao cho tổng số tham số khớp chặt trong khoảng **`±0.14%`** so với `TokenOSCAR 256×32` trên từng cohort:

- **BRCA:** `16,420,996` tham số
- **LUNG:** `18,061,698` tham số
- **COADREAD:** `18,055,044` tham số

---

## 2. Thiết lập thực nghiệm

- **Giao thức:** Matched nested 5-fold CV giống hệt cấu hình chuẩn (`seed = 42`, `MATCHED_MAX_EPOCHS = 120`, inner 20% stratified validation, chọn `first-max` inner-val Macro-F1, `StandardScaler` fit chỉ trên train, retrain trên 100% outer-train, đánh giá 1 lần trên test).
- **Quy mô:** 3 cohort × 5 fold = **15 run / cấu hình** (tổng 60 run mới cho 4 cấu hình đối chứng khớp tham số).
- **Khớp số tham số:** Dùng tìm kiếm nhị phân theo `d_mid` cho từng cấu hình trên từng cohort để giảm thiểu $|P_{\text{variant}} - P_{\text{ref}}|$.
- **Script & dữ liệu thô:**
  - Script: `experiments/run_exp11_param_matched_controls.py`
  - Kết quả thô: `experiments/defense_followup/exp11_param_matched/param_matched_results.json`

### Bảng khớp số tham số chi tiết

| Cấu hình | `d_mid` (BRCA / LUNG / COAD) | Params BRCA (`Δ%`) | Params LUNG (`Δ%`) | Params COADREAD (`Δ%`) |
|---|---|---|---|---|
| `TokenOSCAR 256×32` *(tham chiếu)* | `256 / 256 / 256` | `16,420,996` (`0.00%`) | `18,061,698` (`0.00%`) | `18,055,044` (`0.00%`) |
| `concat_256x32_matched` *(no-attn)* | `257 / 257 / 257` | `16,398,385` (`-0.14%`) | `18,045,497` (`-0.09%`) | `18,038,816` (`-0.09%`) |
| `concat_1x128_matched` *(no-attn)* | `415 / 393 / 393` | `16,429,815` (`+0.05%`) | `18,080,369` (`+0.10%`) | `18,070,016` (`+0.08%`) |
| `mrna_only_mlp_matched` *(no-attn)* | `591 / 628 / 628` | `16,427,957` (`+0.04%`) | `18,051,974` (`-0.05%`) | `18,052,232` (`-0.02%`) |
| `full_1x128_matched` *(1 token + attn)* | `388 / 370 / 370` | `16,422,328` (`+0.01%`) | `18,083,328` (`+0.12%`) | `18,073,596` (`+0.10%`) |

---

## 3. Kết quả thực nghiệm (Matched Nested 5-Fold CV)

### 3.1. Macro-F1 (%, mean ± std)

| Mô hình (khớp số tham số) | BRCA F1 | LUNG F1 | COADREAD F1 | Avg F1 | Δ vs `256×32` | *Bản gốc trước khi khớp tham số* |
|---|---|---|---|---|---|---|
| **`TokenOSCAR 256×32` (tham chiếu)** | **89.93 ± 2.15** | **97.34 ± 1.30** | **89.26 ± 2.55** | **92.18** | **0.00** | *92.18 (16.42M BRCA)* |
| `mrna_only_mlp_matched` | 89.90 ± 3.02 | 95.13 ± 1.52 | **89.26 ± 5.20** | 91.43 | **-0.74** | *90.80 (7.12M BRCA)* |
| `full_1x128_matched` | 88.40 ± 3.54 | 95.58 ± 1.54 | 89.09 ± 3.07 | 91.03 | **-1.15** | *90.76 (11.21M BRCA)* |
| `concat_1x128_matched` | 89.88 ± 2.08 | 95.88 ± 1.44 | 86.56 ± 7.15 | 90.77 | **-1.40** | — |
| `concat_256x32_matched` | 87.06 ± 2.13 | 95.73 ± 1.64 | 84.77 ± 6.03 | 89.19 | **-2.99** | *91.44 (16.33M BRCA)* |

### 3.2. Accuracy (%, mean ± std)

| Mô hình (khớp số tham số) | BRCA Acc | LUNG Acc | COADREAD Acc | Avg Acc | Δ vs `256×32` |
|---|---|---|---|---|---|
| **`TokenOSCAR 256×32` (tham chiếu)** | **91.39 ± 2.18** | **97.36 ± 1.28** | **91.00 ± 2.45** | **93.25** | **0.00** |
| `mrna_only_mlp_matched` | **91.39 ± 2.42** | 95.16 ± 1.50 | 90.98 ± 3.24 | 92.51 | -0.74 |
| `full_1x128_matched` | 90.21 ± 2.49 | 95.60 ± 1.53 | 90.99 ± 2.01 | 92.27 | -0.98 |
| `concat_1x128_matched` | 91.20 ± 2.05 | 95.90 ± 1.43 | 89.69 ± 5.32 | 92.26 | -0.99 |
| `concat_256x32_matched` | 88.84 ± 1.34 | 95.75 ± 1.62 | 86.94 ± 3.57 | 90.51 | -2.74 |

---

## 4. Phân tích & Kết luận

1. **Khớp dung lượng tham số KHÔNG giúp các baseline bắt kịp `TokenOSCAR 256×32`:**
   - Ở cùng ngân sách tham số (`±0.14%`), `TokenOSCAR 256×32` đứng **#1 trên cả 3 cohort** về cả Macro-F1 (`92.18%`) lẫn Accuracy (`93.25%`), vượt các đối chứng khớp tham số từ **`+0.74%` đến `+2.99%` Avg F1**.
2. **Tách rạch ròi vai trò của dung lượng và cơ chế kiến trúc:**
   - Mở rộng `d_mid` giúp `mRNA_only_mlp` (+0.63% Avg F1 so với bản 7.1M cũ) và `1×128 Full` (+0.27% so với bản 11.2M cũ) cải thiện nhẹ trên BRCA và COADREAD, nhưng **vẫn thua rõ rệt trên LUNG** (`-1.76%` đến `-2.21%`) và có phương sai cao hơn trên COADREAD (`±5.20%` ở `mRNA_only_mlp_matched` so với `±2.55%` ở `TokenOSCAR 256×32`).
   - Với `Concat (no-attention)`, việc dồn toàn bộ dung lượng `16.4M–18.1M` vào encoder không có cơ chế chọn lọc/gate làm tăng bất ổn rõ rệt trên tập cỡ mẫu nhỏ COADREAD (`84.77 ± 6.03%` và `86.56 ± 7.15%`).
3. **Kết luận bảo vệ luận văn:** Ưu thế của `TokenOSCAR 256×32` không xuất phát từ việc "nhiều tham số hơn" so với `1×128` hay `mRNA_only_mlp`, mà đến trực tiếp từ **biểu diễn đa token (`T=256`) kết hợp cross-attention và self-attention có cổng (gated fusion)**.
