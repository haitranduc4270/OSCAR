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

*(Chạy đủ cả 6 hoán vị chuỗi mở 2 cạnh trên bản Full có cross gate, self gate,* `pos_embed`*, và* `use_original_for_proteomics = true` *thống nhất trên cả 3 cohort.)*

### Accuracy (%, mean ± std)


| Ordering                                                 | BRCA             | LUNG             | COADREAD         | Avg Acc   | Δ vs Bio |
| -------------------------------------------------------- | ---------------- | ---------------- | ---------------- | --------- | -------- |
| Transcriptomics → proteomics → genomics                  | 91.00 ± 1.30     | 96.77 ± 1.37     | 89.65 ± 2.28     | 92.47     | -0.78    |
| Proteomics → genomics → transcriptomics                  | 90.41 ± 1.99     | 96.34 ± 1.22     | 91.02 ± 4.44     | 92.59     | -0.66    |
| Proteomics → transcriptomics → genomics (reverse)        | **91.98 ± 1.69** | 95.89 ± 1.00     | 90.55 ± 1.66     | 92.80     | -0.45    |
| Transcriptomics → genomics → proteomics (mixed)          | 90.41 ± 2.00     | 96.92 ± 1.57     | **91.46 ± 3.56** | 92.93     | -0.32    |
| Genomics → proteomics → transcriptomics                  | 91.39 ± 2.10     | 96.93 ± 2.18     | 90.57 ± 3.24     | 92.96     | -0.29    |
| **Genomics → transcriptomics → proteomics (TokenOSCAR)** | 91.39 ± 2.18     | **97.36 ± 1.28** | 91.00 ± 2.45     | **93.25** | **0.00** |




### Macro-F1 (%, mean ± std)


| Ordering                                                 | BRCA             | LUNG             | COADREAD         | Avg F1    | Δ vs Bio |
| -------------------------------------------------------- | ---------------- | ---------------- | ---------------- | --------- | -------- |
| Proteomics → genomics → transcriptomics                  | 88.70 ± 2.47     | 96.32 ± 1.23     | 89.06 ± 4.88     | 91.36     | -0.82    |
| Transcriptomics → proteomics → genomics                  | 89.21 ± 1.95     | 96.75 ± 1.38     | 88.52 ± 1.87     | 91.49     | -0.69    |
| Transcriptomics → genomics → proteomics (mixed)          | 88.76 ± 2.91     | 96.91 ± 1.58     | **90.05 ± 3.42** | 91.91     | -0.27    |
| Genomics → proteomics → transcriptomics                  | 90.15 ± 2.62     | 96.91 ± 2.20     | 88.69 ± 3.74     | 91.92     | -0.26    |
| Proteomics → transcriptomics → genomics (reverse)        | **90.85 ± 2.34** | 95.87 ± 1.02     | 89.62 ± 1.35     | 92.11     | -0.07    |
| **Genomics → transcriptomics → proteomics (TokenOSCAR)** | 89.93 ± 2.15     | **97.34 ± 1.30** | 89.26 ± 2.55     | **92.18** | **0.00** |

**Nhận xét:**

- **Thứ tự sinh học (`CNV → mRNA → Proteomics`) đứng hạng #1 trên toàn bộ 6 hoán vị:** Đạt trung bình cao nhất trên cả Accuracy (**93.25%**, hơn 5 hoán vị còn lại từ `+0.29` đến `+0.78%`) lẫn Macro-F1 (**92.18%**, hơn từ `+0.07` đến `+0.82%`).
- **Tác động theo từng cohort:** Một số hoán vị có thể nhỉnh hơn cục bộ trên 1 cohort (như `reverse` cao nhất trên BRCA với `91.98%` Acc / `90.85%` F1, hay `mRNA → CNV → Proteomics` cao nhất trên COADREAD với `91.46%` Acc / `90.05%` F1), nhưng lại đánh đổi ở cohort khác (`reverse` tụt mạnh trên LUNG xuống `95.89%`; `mRNA → CNV → Proteomics` tụt trên BRCA xuống `88.76%` F1). Ngược lại, thứ tự sinh học `CNV → mRNA → Proteomics` cân bằng và ổn định nhất trên cả 3 cohort.
- **Nhóm bắt đầu từ `CNV` (`Genomics`) đều nằm top đầu về Accuracy:** Cả hai thứ tự mở đầu bằng `CNV` (`CNV → mRNA → Proteomics` `93.25%` và `CNV → Proteomics → mRNA` `92.96%`) đều dẫn đầu về Avg Acc, cho thấy dùng thông tin bộ gen làm nguồn ngữ cảnh đầu tiên giúp ổn định quá trình enrich các tầng biểu hiện phía sau.

*(Dữ liệu chi tiết từng fold cho 3 hoán vị bổ sung lưu tại* `lightning_logs/token_oscar_cross_3edge/report.json`*.)*


---



## 4. Sensitivity Study — Ảnh hưởng của cấu hình Token (`T × M`)

Thí nghiệm khảo sát độ nhạy của **TokenOSCAR (full: cross gate + self gate +** `pos_embed`**)** theo số lượng token mỗi omic `T ∈ {1, 64, 128, 256}` và chiều mỗi token `M ∈ {32, 64, 128}` trên lưới lũy thừa cơ số 2 tiêu chuẩn (cùng `d_mid = 256`, `n_heads = 4`, giao thức 5-fold 120 epoch). Cấu hình `1 × 128` đóng vai trò mốc **single-token** trên đúng cùng kiến trúc (tương đương biểu diễn 1 vector/omic của OSCAR gốc).

### Accuracy (%, mean ± std)


| Cấu hình (`T × M`)         | Params (BRCA) | BRCA             | LUNG             | COADREAD         | Avg Acc   | Δ vs `1×128` |
| -------------------------- | ------------- | ---------------- | ---------------- | ---------------- | --------- | ------------ |
| `1 × 128` *(single-token)* | 11.21M        | 89.43 ± 1.44     | 96.92 ± 1.34     | 89.19 ± 1.65     | 91.85     | 0.00         |
| `64 × 32`                  | 11.67M        | 90.60 ± 2.03     | 96.78 ± 1.88     | 90.08 ± 1.84     | 92.49     | +0.64        |
| `128 × 64`                 | 16.63M        | 90.41 ± 1.89     | 96.92 ± 1.26     | **91.44 ± 2.19** | 92.93     | +1.08        |
| `256 × 64`                 | 22.97M        | 91.19 ± 2.16     | 96.77 ± 1.00     | 91.42 ± 2.68     | 93.13     | +1.28        |
| `128 × 32`                 | 13.25M        | 91.19 ± 2.33     | 96.77 ± 1.51     | **91.44 ± 2.19** | 93.14     | +1.29        |
| `256 × 32` *(mặc định)*    | **16.39M**    | **91.39 ± 2.18** | **97.36 ± 1.28** | 91.00 ± 2.45     | **93.25** | **+1.40**    |




### Macro-F1 (%, mean ± std)


| Cấu hình (`T × M`)         | Params (BRCA) | BRCA             | LUNG             | COADREAD         | Avg F1    | Δ vs `1×128` |
| -------------------------- | ------------- | ---------------- | ---------------- | ---------------- | --------- | ------------ |
| `1 × 128` *(single-token)* | 11.21M        | 87.46 ± 2.05     | 96.91 ± 1.35     | 87.90 ± 2.48     | 90.75     | 0.00         |
| `64 × 32`                  | 11.67M        | 89.17 ± 2.77     | 96.75 ± 1.90     | 87.45 ± 2.65     | 91.12     | +0.37        |
| `128 × 64`                 | 16.63M        | 88.61 ± 2.62     | 96.91 ± 1.26     | 89.55 ± 1.47     | 91.69     | +0.94        |
| `256 × 64`                 | 22.97M        | 89.62 ± 2.64     | 96.75 ± 1.01     | 90.17 ± 3.34     | 92.18     | +1.43        |
| `128 × 32`                 | **13.25M**    | **89.94 ± 2.34** | 96.76 ± 1.52     | **90.53 ± 2.04** | **92.41** | **+1.66**    |
| `256 × 32` *(mặc định)*    | **16.39M**    | 89.93 ± 2.15     | **97.34 ± 1.30** | 89.26 ± 2.55     | 92.18     | +1.43        |


**Nhận xét:**

- **Biểu diễn đa token (**`M ∈ {32, 64}`**) vượt trội ổn định so với** `1 × 128`**:** Toàn bộ 5 cấu hình multi-token đều cao hơn `1 × 128` trên trung bình 3 cohort (`+0.64` đến `+1.40%` Acc và `+0.37` đến `+1.66%` Macro-F1), đặc biệt rõ trên hai bài toán phân loại 4 lớp phức tạp (BRCA tăng `+1.15` đến `+2.48%` F1; COADREAD tăng `+1.36` đến `+2.63%` F1 ở nhóm `T ≥ 128`).
- **Tác động lên hai khối attention:** Ở cấu hình `1 × 128`, ma trận cross-attention giữa hai omic suy biến thành kích thước `1 × 1` (trọng số attention luôn bằng `1.0`) và self-attention chỉ thao tác trên chuỗi 4 token (`[CLS]` + 3 omic). Khi tách mỗi omic thành `T ∈ {64, 128, 256}` token với `M ∈ {32, 64}`, cross-attention thực sự học được tương tác chọn lọc `T × T` xuyên omic và 2 lớp Transformer self-attention khai thác được ngữ cảnh chuỗi dài (`3T + 1` token).
- **Không phụ thuộc vào "magic number":** Các cấu hình `128 × 32`, `256 × 32` và `256 × 64` đều hội tụ rất sát nhau (`93.13–93.25%` Acc, `92.18–92.41%` F1). Thiết lập mặc định `256 × 32` đạt Avg Acc cao nhất (`93.25%`) trong khi nhẹ hơn `256 × 64` khoảng `6.6M` tham số.

*(Dữ liệu chi tiết từng fold lưu tại* `lightning_logs/token_oscar_sweep_v2/{t1_m128,t64_m32,t128_m32,t128_m64,t256_m32,t256_m64}/report.json`*.)*

### Đối chiếu tác dụng của 2 khối Attention: `1 × 128` vs `256 × 32`

Để kiểm chứng trực tiếp rằng biểu diễn đa token giúp hai khối attention phía sau thực sự phát huy tác dụng, ta chạy cùng nhóm ablation loại bỏ thành phần trên `1 × 128` (single-token) và so sánh độ thay đổi (`Δ vs Full`) với `256 × 32`:


| Biến thể                                          | `1 × 128` Avg Acc (Δ vs Full) | `256 × 32` Avg Acc (Δ vs Full) | `1 × 128` Avg F1 (Δ vs Full) | `256 × 32` Avg F1 (Δ vs Full) |
| ------------------------------------------------- | ----------------------------- | ------------------------------ | ---------------------------- | ----------------------------- |
| Concatenation *(tắt cả 2 attention)*              | 92.44 (+0.59)                 | 92.00 (-1.25)                  | 91.15 (+0.40)                | 91.44 (-0.74)                 |
| Without self-attention *(chỉ cross + gate)*       | **92.85 (+1.00)**             | 91.56 (-1.69)                  | **91.69 (+0.94)**            | 90.80 (-1.38)                 |
| Without cross-attention *(chỉ self + gate + pos)* | 91.40 (-0.45)                 | 92.69 (-0.56)                  | 90.03 (-0.72)                | 91.38 (-0.80)                 |
| **Full *(cross + self + 2 gates + pos)***         | **91.85 (0.00)**              | **93.25 (0.00)**               | **90.75 (0.00)**             | **92.18 (0.00)**              |


- **Ở** `1 × 128`**, hai khối attention không phát huy tác dụng:** bản `Full (1 × 128)` thấp hơn cả `Concatenation` (`91.85` vs `92.44` Acc; `90.75` vs `91.15` F1), và việc bỏ hẳn 2 lớp Transformer self-attention lại tăng `+1.00%` Acc / `+0.94%` F1 (`92.85` / `91.69`) do self-attention trên chuỗi 4 token bị thừa tham số.
- **Ở** `256 × 32`**, cả hai khối attention trở thành động lực cải thiện chính:** bản `Full (256 × 32)` vượt `Concatenation` (`+1.25%` Acc / `+0.74%` F1) và vượt mốc cao nhất của toàn bộ nhóm `1 × 128`; bỏ self-attention làm tụt `-1.69%` Acc / `-1.38%` F1, còn bỏ cross-attention làm tụt `-0.56%` Acc / `-0.80%` F1.

*(Dữ liệu chi tiết từng fold cho* `1 × 128` *lưu tại* `lightning_logs/token_oscar_ablation_1x128/{concat,no_cross_self_gate_pos,no_self_attn}/report.json`*.)*

### Kiểm chứng tính tổng quát của `OmicToTokens (256 × 32)` trên các baseline attention (`MOT` & `MoXGATE`)

Để kiểm tra xem mức cải thiện của **TokenOSCAR (`256 × 32`)** có chỉ đến từ bước chiếu đa token hay không, ta thay trực tiếp lớp chiếu single-vector ở đầu vào của hai baseline dựa trên multi-head attention là **MOT** và **MoXGATE** bằng đúng cùng mô-đun `OmicToTokens (T = 256, M = 32, d_mid = 256)`, giữ nguyên toàn bộ kiến trúc attention, hàm loss, optimizer và giao thức 5-fold matched phía sau (`train_baselines_token.py`):

#### Accuracy (%, mean ± std)

| Model | BRCA | LUNG | COADREAD | Avg Acc | Δ vs Orig |
|---|---|---|---|---|---|
| MOT *(orig)* | 90.80 ± 2.11 | 95.90 ± 1.51 | 90.11 ± 2.19 | 92.27 | 0.00 |
| **MOT + `256 × 32`** | **91.58 ± 1.83** | 95.60 ± 1.46 | 90.10 ± 3.31 | **92.43** | **+0.16** |
| MoXGATE *(orig)* | 89.23 ± 1.65 | 96.04 ± 0.75 | 88.30 ± 2.93 | 91.19 | 0.00 |
| **MoXGATE + `256 × 32`** | 87.09 ± 1.54 | 96.04 ± 1.28 | **89.65 ± 4.15** | 90.93 | -0.27 |
| **TokenOSCAR (`256 × 32`)** | 91.39 ± 2.18 | **97.36 ± 1.28** | **91.00 ± 2.45** | **93.25** | — |

#### Macro-F1 (%, mean ± std)

| Model | BRCA | LUNG | COADREAD | Avg F1 | Δ vs Orig |
|---|---|---|---|---|---|
| MOT *(orig)* | 89.28 ± 3.19 | 95.87 ± 1.52 | 87.27 ± 2.13 | 90.81 | 0.00 |
| **MOT + `256 × 32`** | **89.86 ± 2.89** | 95.58 ± 1.48 | **89.53 ± 2.75** | **91.65** | **+0.85** |
| MoXGATE *(orig)* | 86.82 ± 2.69 | 96.02 ± 0.76 | 86.09 ± 2.86 | 89.64 | 0.00 |
| **MoXGATE + `256 × 32`** | 84.27 ± 2.63 | 96.02 ± 1.29 | **89.40 ± 4.60** | **89.90** | **+0.25** |
| **TokenOSCAR (`256 × 32`)** | **89.93 ± 2.15** | **97.34 ± 1.30** | 89.26 ± 2.55 | **92.18** | — |

**Nhận xét:**

- **Cách chiếu `OmicToTokens (256 × 32)` thực sự mang lại lợi ích, nhưng không tự tạo ra hiệu năng vượt trội một cách độc lập:** Khi gắn cùng bộ projection `256 × 32`, `MOT` tăng rõ về Macro-F1 (`+0.85%`, đặc biệt COADREAD `+2.26%`), còn `MoXGATE` chỉ tăng nhẹ F1 (`+0.25%`) và giảm nhẹ Acc (`-0.27%`, tụt mạnh trên BRCA `-2.55%` F1).
- **Cần kiến trúc phía sau đủ khả năng khai thác chuỗi token dài:** `MOT` tận dụng tốt hơn `MoXGATE` vì Transformer Encoder 2 lớp có đầy đủ Pre-LN, Residual và FFN; ngược lại, `JointCrossAttention` của `MoXGATE` không có Residual/LayerNorm/FFN và bị phạt Frobenius `λ₂‖W_c‖_F²`, nên dễ bị over-smoothing trên chuỗi 768 token.
- **Lợi thế của TokenOSCAR đến từ sự kết hợp đồng bộ giữa projection và kiến trúc fusion:** Ngay cả khi đặt trên cùng mặt bằng `256 × 32`, **TokenOSCAR** vẫn vượt cả `MOT + 256 × 32` (`+0.82%` Avg Acc / `+0.53%` Avg F1) lẫn `MoXGATE + 256 × 32` (`+2.32%` Avg Acc / `+2.28%` Avg F1), khẳng định hiệu năng không chỉ đến từ tokenization mà còn cần **ordered cross-attention**, **gated fusion** và **self-attention có gate** để khai thác hiệu quả biểu diễn đa token.

*(Dữ liệu chi tiết từng fold lưu tại* `lightning_logs/baselines_token_256x32/{brca,lung,coadread}_report.json` *và* `lightning_logs/baselines_token_256x32/table.md`*.)*

---



## 5. Table 5.4 — Enrichr Over-Representation on BRCA (TokenOSCAR `256×32`)

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



## 6. Nhận xét chính

1. **Vai trò của Self-Attention:** Bỏ self-attention (`Without self-attention` và `Without gate & self-attention`) làm giảm mạnh nhất trên trung bình cả 3 cohort (Avg F1 tụt `-1.64` và `-2.18`), đặc biệt nặng trên COADREAD (`-4.62` và `-8.30` F1).
2. **Vai trò của Gated Fusion:** Trên COADREAD (N = 222, 4 lớp CMS), bỏ gate làm tụt `-4.23` F1 / `-2.67` Acc và tăng độ dao động (std F1 từ `2.55` lên `3.94`). Trên BRCA, bỏ gate gần như hòa về điểm trung bình (`+0.28` F1 / `+0.19` Acc) nhưng kém ổn định hơn (std F1 `2.75` vs `2.17`), do `TokenCrossAttention` đã có sẵn residual connection.
3. **Vai trò của Cross-Attention:** Biến thể `Without cross-attention` (chỉ dùng Transformer self-attention) đạt Avg Acc cao nhất (**93.08%**, đặc biệt mạnh trên LUNG `96.77%`), nhưng trên COADREAD lại giảm F1 (`88.81%` vs `91.34%`) và dao động rất lớn (`std 6.31` vs `2.55`). Nhờ đó `TokenOSCAR (full)` vẫn giữ Avg F1 cao nhất trong nhóm component ablation (**91.78%**).
4. **Ảnh hưởng của thứ tự (Table 5.3):** Khác với OSCAR `1×128` trong `main.pdf`, trên TokenOSCAR `256×32`, thứ tự sinh học `CNV → mRNA → Proteomics` chỉ thắng rõ trên **COADREAD** (`92.33%` Acc / `91.34%` F1). Trên **BRCA**, cả `reverse` và `mixed` đều nhỉnh hơn nhẹ (`91.58%` vs `91.00%` Acc). Trên **LUNG**, `TokenOSCAR (full)` bị kéo xuống do fold 1 dừng sớm ở `chosen_epoch = 1` (`87.59%` Acc).

---



## 7. Lệnh tái lập

```bash
# Chạy toàn bộ 8 biến thể ablation (Table 5.2 mở rộng & Table 5.3) trên 3 cohort
uv run python train_token_oscar_ablation.py --cohort all --variants all

# Chạy sensitivity study (T x M) trên bản TokenOSCAR full
for cfg in "1 128 t1_m128" "64 32 t64_m32" "128 32 t128_m32" "128 64 t128_m64" "256 32 t256_m32" "256 64 t256_m64"; do
  set -- $cfg
  uv run python train_token_oscar_matched.py --cohort all --num-tokens $1 --token-dim $2 --out-dir lightning_logs/token_oscar_sweep_v2/$3
done

# Đối chiếu ablation thành phần trên 1x128
for v in concat no_self_attn no_cross_self_gate_pos; do
  uv run python train_token_oscar_ablation.py --cohort all --variants $v --num-tokens 1 --token-dim 128 --out-dir lightning_logs/token_oscar_ablation_1x128/$v
done

# Kiểm chứng gắn OmicToTokens (256x32) vào MOT và MoXGATE
uv run python train_baselines_token.py --cohort all --models mot,moxgate

# Chạy Enrichr (Table 5.4) cho TokenOSCAR (256x32)
uv run --with gseapy python metrix/biomarker_step3_enrichment.py \
  --consensus-csv attention_plots/token_oscar_step1_consensus/consensus_features.csv \
  --de-csv attention_plots/token_oscar_step2_de/consensus_with_de.csv \
  --output-dir attention_plots/token_oscar_step3_enrichment \
  --class-names LumA,LumB,Basal,HER2
```

Dữ liệu chi tiết từng fold được lưu tại `ablation_study/results.json`; bảng Enrichr đạt FDR < 0.05 lưu tại `ablation_study/table5_4_enrichment_sig.csv`.