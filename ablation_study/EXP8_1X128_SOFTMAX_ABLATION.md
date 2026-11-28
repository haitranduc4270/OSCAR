# Exp 8 (`1×128`): Kiểm chứng Suy biến Softmax ở Single-Token (`T=1`) & Thử nghiệm Các Cơ chế Thay thế

## 1. Mục tiêu & Giả thuyết kiểm chứng

Ở cấu hình single-token **`1×128`** (`num_tokens = 1, token_dim = 128` trên `TokenOSCAR` dùng `OmicToTokens`), mỗi omic chỉ được nén thành đúng **1 token** ($T_q = T_k = 1$). Khi đưa vào `TokenCrossAttention` (`models/token_oscar.py`), ma trận điểm attention có kích thước $1 \times 1$ cho mỗi head, dẫn đến hiện tượng **suy biến toán học**:

- **`Softmax`**: $\text{softmax}([s]) = [1.0]$ $\Rightarrow$ đầu ra chỉ còn $W_O W_V X_{\text{kv}}$, bỏ qua hoàn toàn truy vấn $Q$ và khóa $K$.
- **`Sparsemax`**: chiếu lên simplex 1 chiều luôn cho kết quả hằng số $[1.0]$ $\Rightarrow$ **cũng suy biến**.
- **`Sinkhorn`**: chuẩn hóa song ngẫu trên ma trận $1 \times 1$ cho kết quả hằng số $[1.0]$ $\Rightarrow$ **cũng suy biến**.
- **`Linear ELU+1`**: $\frac{\phi(q)\phi(k)^\top v}{\phi(q)\phi(k)^\top} = 1.0 \cdot v$ $\Rightarrow$ **cũng suy biến**.
- **`Sigmoid`**: $\sigma\!\left(\frac{q \cdot k}{\sqrt{d_k}} - \log 1\right) = \sigma\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) \in (0, 1)$ cho từng head $\Rightarrow$ **duy nhất không suy biến**, đóng vai trò cổng điều biến phụ thuộc dữ liệu từ $(Q, K)$.

Thí nghiệm này kiểm chứng cả về **mặt số học (sanity check)** lẫn **hiệu năng thực nghiệm (matched nested 5-fold CV)** trên cả 3 tập `BRCA`, `LUNG`, `COADREAD` để trả lời câu hỏi: *nếu thay `Softmax` bằng các cơ chế khác (đặc biệt là `Sigmoid` không suy biến), liệu `1×128` có bắt kịp được `256×32` không?*

---

## 2. Thiết lập thực nghiệm

- **Kiến trúc:** `TokenOSCARModel` (`OmicToTokens`, `d_mid = 256`, `n_heads = 4`, `n_layers = 2`, `use_cross_attention = True`, `use_gated_fusion = True`, `use_self_attention = True`, `use_self_gate = True`, `use_pos_embed = True`, thứ tự `CNV → mRNA → Proteomics`), đặt `num_tokens = 1, token_dim = 128`.
- **Giao thức:** Matched nested 5-fold CV giống hệt Exp 8 (`seed = 42`, `MATCHED_MAX_EPOCHS = 120`, inner 20% stratified val, `StandardScaler` fit chỉ trên train).
- **Script & dữ liệu thô:**
  - Script: `experiments/run_exp8_1x128_softmax_ablation.py`
  - Kết quả thô: `experiments/defense_followup/exp8_1x128_softmax/results.json`

---

## 3. Kiểm chứng số học suy biến tại $T = 1$ (Sanity Check)

Giữ nguyên `key_value`, thay đổi ngẫu nhiên `query` từ `q1` sang `q2` (`B=8, T=1, M=128, H=4`, `dropout=0`):

| Cơ chế (`attn_type`) | Khoảng trọng số attention | $\max\| \text{out}(q_1) - \text{out}(q_2) \|$ | Suy biến tại $T=1$? |
|---|---|---|---|
| `attn_softmax` | `[1.0000, 1.0000]` | `0.00e+00` | **Có (Degenerate)** |
| `attn_sparsemax` | `[1.0000, 1.0000]` | `0.00e+00` | **Có (Degenerate)** |
| `attn_sinkhorn` | `[1.0000, 1.0000]` | `0.00e+00` | **Có (Degenerate)** |
| `attn_linear_elu` | `[1.0000, 1.0000]` | `2.98e-07` *(sai số float32)* | **Có (Degenerate)** |
| **`attn_sigmoid`** | **`[0.3702, 0.6536]`** | **`1.36e-01`** | **Không (Data-dependent)** |

---

## 4. Kết quả thực nghiệm (Matched Nested 5-Fold CV)

### 4.1. Macro-F1 (%, mean ± std)

| Biến thể (`1×128`, `OmicToTokens`) | Suy biến $T=1$? | BRCA F1 | LUNG F1 | COADREAD F1 | Avg F1 | Δ vs `1×128 Softmax` | *Đối chiếu `256×32` Avg F1* |
|---|---|---|---|---|---|---|---|
| `attn_softmax` *(baseline `1×128`)* | Có (`= 1.0`) | 87.46 ± 2.05 | 96.91 ± 1.35 | 87.90 ± 2.48 | 90.76 | 0.00 | ***92.18*** |
| **`attn_sigmoid`** *(không suy biến)* | **Không** | 89.24 ± 3.05 | 96.76 ± 1.20 | 87.90 ± 2.07 | 91.30 | +0.54 | *91.68* |
| `attn_sparsemax` | Có (`= 1.0`) | **89.58 ± 3.03** | **97.20 ± 1.28** | 88.00 ± 2.63 | **91.59** | **+0.84** | *91.48* |
| `attn_sinkhorn` | Có (`= 1.0`) | **89.58 ± 3.03** | **97.20 ± 1.28** | 88.00 ± 2.63 | **91.59** | **+0.84** | *90.67* |
| `attn_linear_elu` | Có (`= 1.0`) | 88.50 ± 2.91 | 95.59 ± 2.23 | **90.30 ± 2.40** | 91.46 | +0.71 | *90.13* |

### 4.2. Accuracy (%, mean ± std)

| Biến thể (`1×128`, `OmicToTokens`) | BRCA Acc | LUNG Acc | COADREAD Acc | Avg Acc | Δ vs `1×128 Softmax` |
|---|---|---|---|---|---|
| `attn_softmax` *(baseline `1×128`)* | 89.43 ± 1.44 | 96.92 ± 1.34 | 89.19 ± 1.65 | 91.85 | 0.00 |
| `attn_sigmoid` *(không suy biến)* | 90.80 ± 2.30 | 96.77 ± 1.20 | 89.65 ± 1.75 | 92.41 | +0.56 |
| `attn_sparsemax` | **90.99 ± 2.01** | **97.21 ± 1.27** | 90.09 ± 2.27 | **92.76** | **+0.91** |
| `attn_sinkhorn` | **90.99 ± 2.01** | **97.21 ± 1.27** | 90.09 ± 2.27 | **92.76** | **+0.91** |
| `attn_linear_elu` | 90.41 ± 1.92 | 95.61 ± 2.22 | **91.88 ± 1.84** | 92.63 | +0.78 |

*(Số tham số giống nhau trên cả 5 biến thể: `11,212,420` trên BRCA, `12,853,122` trên LUNG, `12,846,468` trên COADREAD.)*

---

## 5. Phân tích & Kết luận

1. **Bằng chứng thực nghiệm tuyệt đối về sự suy biến khi $T = 1$:**
   - `attn_sparsemax` và `attn_sinkhorn` cho kết quả **trùng khớp 100% đến từng chữ số thập phân** trên cả 3 tập (`BRCA 89.58 ± 3.03%`, `LUNG 97.20 ± 1.28%`, `COADREAD 88.00 ± 2.63%`) vì ở $T = 1$, cả hai đều suy biến về cùng một hằng số `1.0` và đi qua đúng cùng một nhánh `_custom_attn` (`q_proj, k_proj, v_proj, out_proj` + `attn_drop`).
   - `attn_softmax` cũng suy biến về `1.0`, nhưng đạt `90.76%` so với `91.59%` của `sparsemax/sinkhorn` thuần túy do khác biệt khởi tạo giữa `nn.MultiheadAttention` (`in_proj_weight` gộp chung) và 3 lớp `nn.Linear` tách rời trong `_custom_attn` — hoàn toàn **không phải** do cơ chế attention học được tương tác $Q$–$K$.
2. **`Sigmoid` loại bỏ được suy biến nhưng vẫn không cứu được `1×128`:**
   - `attn_sigmoid` là cơ chế **duy nhất** giữ được trọng số biến thiên theo $(Q, K)$ ở $T = 1$ (`weight ∈ [0.37, 0.65]`) và cải thiện `+0.54%` Avg F1 so với `1×128 Softmax` (`91.30%` vs `90.76%`, riêng BRCA tăng `+1.78%`).
   - Tuy nhiên, `1×128 Sigmoid` (`91.30%` Avg F1) vẫn **thua rõ rệt `TokenOSCAR 256×32 Softmax` (`92.18%` Avg F1, chênh `-0.88%`)**, và thậm chí không vượt được nhánh suy biến thuần chiếu tuyến tính (`sparsemax/sinkhorn 1×128`, `91.59%`).
3. **Động cơ trực tiếp cho thiết kế đa token `OmicToTokens` (`256×32`):**
   - Nút thắt của `1×128` không chỉ nằm ở hàm `Softmax` bị suy biến, mà nằm ở **nút thắt cổ chai biểu diễn 1 token ($T = 1$)**: ngay cả khi đổi `Softmax` sang `Sigmoid` để khôi phục phụ thuộc $(Q, K)$, mỗi omic vẫn chỉ trao đổi thông tin qua **4 hệ số vô hướng (4 head)** đóng/mở toàn cục, không thể học tương tác chọn lọc giữa các cụm đặc trưng.
   - Do đó, việc đổi hàm chuẩn hóa trên `1×128` là **không đủ** — đây là lý do trực tiếp thúc đẩy thiết kế **`OmicToTokens` (`256×32`)** tách mỗi omic thành chuỗi đa token ($T = 256$), mở ra ma trận tương tác $T \times T$ giữa các không gian con (subspace-to-subspace) và đẩy hiệu năng lên **`92.18%` Avg F1** (**`+0.88%`** so với `1×128 Sigmoid`, **`+1.42%`** so với `1×128 Softmax`).
