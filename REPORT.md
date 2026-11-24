# TokenOSCAR — Báo cáo Thực nghiệm (Matched 5-Fold)

## 1. Động lực & Kiến trúc

Trong OSCAR gốc (`1×128`), mỗi omic (CNV, mRNA, Proteomics) bị nén thành **1 vector duy nhất** có chiều 128 trước khi đưa vào cross-attention. Khi độ dài chuỗi bằng 1, phép `softmax(QKᵀ/√d)` luôn bằng `1`, dẫn tới attention suy biến (gradient theo `Q`, `K` bằng 0 ở SDP attention; đầu ra chỉ phụ thuộc `V`).

**TokenOSCAR (`models/token_oscar.py`)** giữ mỗi omic dưới dạng chuỗi `T` token có chiều `M` (`T > 1, M > 1`) xuyên suốt toàn bộ quá trình hợp nhất:

1. **Omic-to-Tokens (`OmicToTokens`):** chiếu phân tách `(B, D) → Linear(D, 256) → LN → GELU → Dropout → Linear(256, T·M) → (B, T, M)`.
2. **Ordered Token Cross-Attention (`TokenCrossAttention`):** multi-head cross-attention thực sự (`T × T`, 4 heads, Pre-LN + FFN×4) theo thứ tự sinh học `CNV → mRNA → Proteomics`.
3. **Token-wise Gated Fusion (`TokenGatedFusion`):** kết hợp đặc trưng gốc và đặc trưng sau cross-attention tại từng token bằng cổng sigmoid.
4. **Multi-Omic Self-Attention Fusion:** cộng modality embedding, ghép 3 chuỗi token với `[CLS]` (`3T + 1` token), đi qua Transformer Encoder 2 lớp, rồi phân loại trên `[CLS]`.

Cấu hình mặc định đã chốt: **`T = 256, M = 32`** (`d_mid = 256`, `n_heads = 4`, chuỗi fusion 769 token, ~16.39M tham số trên BRCA).

---

## 2. Giao thức Đánh giá (Matched 5-Fold Protocol)

Tất cả mô hình được đánh giá trên cùng một luồng (`train_token_oscar_matched.py`, `train_baselines_matched.py`):

1. **Outer folds:** 5 fold chia sẵn tại `csv/processed/{BRCA,LUNG,COADREAD}/5-fold/fold_{1..5}`. File `*_val.csv` của mỗi fold là tập test độc lập.
2. **Inner split:** tách 20% stratified từ tập train của fold (`seed=42`) làm inner-val.
3. **Chuẩn hóa:** `StandardScaler` chỉ fit trên phần train ở từng giai đoạn.
4. **Chọn epoch:** train tối đa 120 epoch, chọn **epoch đầu tiên** đạt inner-val macro-F1 cao nhất. Không tune thêm siêu tham số khác trong luồng đánh giá.
5. **Retrain & Test:** khởi tạo lại với `seed=42`, train trên 100% tập train của fold đúng số epoch đã chọn, chấm **1 lần duy nhất** trên tập test của fold.
6. **Báo cáo:** `mean ± std` của Macro-F1 và Accuracy (%) trên 5 fold.

*(Các baseline đồ thị transductive — MoGCN, DeepMoIC, MOGONET — xây đồ thị bệnh nhân trên train + eval nhưng nhãn và loss chỉ dùng phần train.)*

---

## 3. So sánh với Baselines

Đơn vị: **Macro-F1 / Accuracy (%, mean ± std)**.

| Method | BRCA (4 lớp) | LUNG (2 lớp) | COADREAD (4 lớp) | Avg F1 | Δ Avg F1 vs TokenOSCAR |
|---|---|---|---|---|---|
| **TokenOSCAR (256×32)** | **89.31±2.17 / 91.00±1.69** | 94.70±3.76 / 94.73±3.73 | 91.34±2.55 / 92.33±2.73 | **91.78** | — |
| OSCAR (1×128)\* | 87.01±3.41 / 88.85±2.58 | 95.72±1.70 / 95.75±1.69 | **93.43±2.96 / 94.13±3.09** | **92.05** | +0.27 |
| MOT | 89.28±3.19 / 90.80±2.11 | 95.87±1.52 / 95.90±1.51 | 87.27±2.13 / 90.11±2.19 | 90.81 | -0.97 |
| MoXGATE (paper) | 86.82±2.69 / 89.23±1.65 | **96.02±0.76 / 96.04±0.75** | 86.09±2.86 / 88.30±2.93 | 89.64 | -2.14 |
| MoGCN | 81.29±2.08 / 85.33±1.34 | 94.10±2.47 / 94.14±2.45 | 84.39±3.32 / 86.95±2.13 | 86.59 | -5.19 |
| MOGONET | 74.55±2.71 / 82.39±0.62 | 93.45±2.61 / 93.55±2.53 | 91.37±2.70 / 92.33±1.16 | 86.46 | -5.32 |
| moBRCA-net | 83.05±3.20 / 85.91±1.58 | 95.43±1.58 / 95.46±1.56 | 80.60±5.73 / 83.30±4.02 | 86.36 | -5.42 |
| DeepMoIC | 81.17±2.09 / 85.13±1.43 | 92.92±2.61 / 92.97±2.59 | 80.96±10.01 / 86.08±5.65 | 85.02 | -6.76 |

\* *OSCAR (1×128) lấy từ lần chạy `table51` trước đó (`lightning_logs/{brca,lung,coadread}_table51_matched/`). Sáu baseline còn lại chạy từ `train_baselines_matched.py` (`lightning_logs/baselines_matched/`).*

### Nhận xét chính
- **BRCA:** TokenOSCAR (`256×32`) đạt cao nhất cả F1 (**89.31%**) và Accuracy (**91.00%**), vượt OSCAR `1×128` (+2.30 F1) và ổn định hơn MOT (std 2.17 vs 3.19).
- **COADREAD:** đạt **91.34%** F1, ngang MOGONET (91.37%), vượt rõ MOT (+4.07) và MoXGATE (+5.25), chỉ sau OSCAR `1×128` (93.43%).
- **LUNG:** đạt **94.70%** F1 (thấp hơn MoXGATE 96.02% và MOT 95.87% do 1 fold dao động, std 3.76).
- **Tổng thể:** TokenOSCAR (`256×32`) vượt toàn bộ 6 baseline bên ngoài (**+0.97** đến **+6.76** Avg F1) và gần tương đương OSCAR `1×128` (-0.27 Avg F1).

---

## 4. Khảo sát kích thước Token (`T × M`)

| Cấu hình (`T × M`) | BRCA F1 | LUNG F1 | COADREAD F1 | Avg F1 | Params (BRCA) | Peak VRAM |
|---|---|---|---|---|---|---|
| `128 × 32` | 88.73±3.07 | 96.17±1.08 | 90.25±2.47 | 91.72 | 13.23M | 803 MB |
| `128 × 64` | 88.06±3.09 | 96.17±1.70 | **91.37±3.81** | **91.86** | 16.58M | 907 MB |
| `128 × 128` | **89.48±2.85** | **96.90±1.36** | 85.25±2.58 | 90.54 | 23.65M | 1,124 MB |
| **`256 × 32` (default)** | 89.31±2.17 | 94.70±3.76 | 91.34±2.55 | 91.78 | 16.39M | 2,365 MB |
| `256 × 64` | 88.07±2.52 | 96.46±0.72 | 88.47±3.01 | 91.00 | 22.90M | 2,574 MB |

---

## 5. Lệnh tái lập

```bash
# TokenOSCAR (mặc định 256x32) trên cả 3 cohort
uv run python train_token_oscar_matched.py --cohort all

# Tùy biến T, M
uv run python train_token_oscar_matched.py --cohort all --num-tokens 128 --token-dim 64 \
  --out-dir lightning_logs/token_oscar_sweep/t128_m64

# 6 baseline trên cả 3 cohort
uv run python train_baselines_matched.py --cohort all --models all
```

## 6. Giới hạn đã biết

- Các số OSCAR `1×128` lấy từ script `train_table51_matched.py` cũ (không còn trên nhánh này).
- Config MoXGATE cho LUNG (`configs/config_moxgate_lung.yaml`) và COADREAD (`configs/config_moxgate_crc.yaml`) được tái dựng từ config BRCA của paper (chỉ đổi đường dẫn, `num_classes` và `log_name`); kết quả chạy lại khớp chính xác số `table51` cũ.
- Các baseline không được tune siêu tham số riêng theo từng cohort; chỉ tune `chosen_epoch` trên inner-val giống nhau cho mọi mô hình.
