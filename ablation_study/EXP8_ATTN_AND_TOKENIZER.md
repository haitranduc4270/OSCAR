# Exp 8: Thử nghiệm Thay thế `Softmax` trong Cross-Attention & Tinh chỉnh Tokenizer `OmicToTokens`

## 1. Tại sao thực hiện thử nghiệm này?

Thử nghiệm này xuất phát trực tiếp từ hai câu hỏi kỹ thuật cốt lõi sau khi kiểm toán kiến trúc `TokenOSCAR (256×32)` ở **Exp 7a** và **Q1**:

1. **Vấn đề độ phẳng của Softmax Cross-Attention:** Kết quả kiểm toán trên 5 checkpoint BRCA (Exp 7a) cho thấy ma trận cross-attention dùng `Softmax` chuẩn có normalized entropy rất cao (**`0.9802`** cho `CNV → mRNA` và **`0.9854`** cho `mRNA → Proteomics`, so với uniform = `1.0`). Điều này đặt ra câu hỏi: *liệu thay `Softmax` bằng các cơ chế chuẩn hóa khác — độc lập không cạnh tranh (`Sigmoid`), thưa có trọng số bằng đúng 0 (`Sparsemax`), tuyến tính hóa bằng kernel (`Linear ELU+1`), hoặc song ngẫu hai chiều (`Sinkhorn / Optimal Transport`) — có giúp attention sắc nét hơn, lọc nhiễu CNV tốt hơn và cải thiện Macro-F1 không?*
2. **Kiểm tra lại thiết kế Tokenizer `OmicToTokens`:** Dù Exp 7c đã chứng minh `OmicToTokens` (`92.18%`) vượt xa `FeatureChunkToTokens` kiểu ViT (`65.17%–84.63%`), vẫn còn 2 nghi vấn cần kiểm chứng dứt điểm:
   - Thêm `LayerNorm` ngay sau bước reshape `(B, 256, 32)` (`tok_layernorm`) có giúp ổn định biên độ token trước khi vào cross-attention không?
   - Tăng bottleneck `d_mid` từ `256` lên `512` (`tok_dmid512`) có mở rộng thêm thông tin không hay `d_mid = 256` đã đạt điểm bão hòa?

---

## 2. Thiết lập thực nghiệm

- **Giao thức:** Matched nested 5-fold CV giống hệt bảng chính (`seed = 42`, inner 20% stratified validation chọn `first-max` Macro-F1 trong budget 120 epoch, `StandardScaler` fit chỉ trên train, retrain 100% outer-train, đánh giá 1 lần trên test).
- **Tập dữ liệu:** `BRCA` ($N=511$, 4 lớp), `LUNG` ($N=682$, 2 lớp), `COADREAD` ($N=222$, 4 lớp).
- **Các biến thể đối chứng** (cài đặt trong `models/token_oscar.py`, script chạy `experiments/run_exp8_attn_and_tokenizer.py`, dữ liệu thô `experiments/defense_followup/exp8_attn_tokenizer/exp8_results.json`):
  - `attn_sigmoid`: $\sigma\!\left(\frac{QK^\top}{\sqrt{d_k}} - \log T\right)V$ (thêm bias $-\log T$ chuẩn *Sigmoid Attention* để tránh nổ biên độ khi $T=256$).
  - `attn_sparsemax`: Chiếu trực tiếp lên simplex (Martins & Astudillo, 2016), ép các trọng số nhỏ về đúng `0`.
  - `attn_linear_elu`: Kernel attention tuyến tính $\phi(x)=\text{ELU}(x)+1$, tính $\frac{\phi(Q)(\phi(K)^\top V)}{\phi(Q)\sum \phi(K)^\top + \epsilon}$.
  - `attn_sinkhorn`: 5 vòng Sinkhorn-Knopp trong log-space tạo ma trận attention gần song ngẫu.
  - `tok_layernorm`: Thêm `nn.LayerNorm(32)` ngay sau reshape `(B, 256, 32)` trong `OmicToTokens`.
  - `tok_dmid512`: Tăng chiều bottleneck `d_mid` từ `256` lên `512`.

---

## 3. Kết quả thực nghiệm (5-Fold Macro-F1 & Accuracy)

| Biến thể | Params (BRCA) | BRCA F1 | LUNG F1 | COADREAD F1 | Avg F1 | Δ vs Default |
|---|---|---|---|---|---|---|
| **`TokenOSCAR (Softmax, d_mid=256)`** | 16,420,996 | **89.93 ± 2.15%** | **97.34 ± 1.30%** | **89.26 ± 2.55%** | **92.18%** | — |
| `attn_sigmoid` | 16,420,996 | 89.62 ± 2.43% | 96.17 ± 1.27% | 89.26 ± 3.31% | 91.68% | -0.50% |
| `tok_layernorm` | 16,421,188 | 89.34 ± 2.51% | 97.20 ± 0.73% | 88.28 ± 1.35% | 91.61% | -0.57% |
| `attn_sparsemax` | 16,420,996 | 89.10 ± 1.22% | 96.62 ± 0.99% | 88.73 ± 1.70% | 91.48% | -0.70% |
| `tok_dmid512` | 32,718,212 | 89.49 ± 2.69% | 96.91 ± 0.98% | 86.05 ± 6.16% | 90.82% | -1.36% |
| `attn_sinkhorn` | 16,420,996 | 87.41 ± 1.55% | 96.32 ± 1.54% | 88.29 ± 2.67% | 90.67% | -1.51% |
| `attn_linear_elu` | 16,420,996 | 89.44 ± 1.34% | 96.31 ± 0.82% | 84.63 ± 10.83% | 90.13% | -2.05% |

---

## 4. Nhận xét & Kết luận

1. **`Softmax` mặc định vẫn là lựa chọn tốt nhất:** Dù `Softmax` có entropy chuẩn hóa cao (`0.980–0.985`), cấu hình mặc định **`TokenOSCAR (Softmax, d_mid=256)` vẫn đạt #1 trên cả 3 tập (`92.18%` Avg F1)**. Điều này cho thấy với 256 latent token biểu diễn các *subspace* toàn cục từ `OmicToTokens`, cơ chế trộn phân tán mềm (soft distributed mixing) của `Softmax` phù hợp hơn so với ép chọn cứng vài token.
2. **Đánh giá các phương án thay thế `Softmax`:**
   - `Sigmoid` (`91.68%`) và `Sparsemax` (`91.48%`) bám sát nhất; riêng `Sparsemax` cho **độ ổn định cực cao giữa các fold** (`std` chỉ `±1.22%` BRCA, `±0.99%` LUNG, `±1.70%` COADREAD), đáng cân nhắc nếu ưu tiên attention thưa để diễn giải.
   - `Sinkhorn` (`90.67%`) quá gò bó khi ép ghép cặp song ngẫu 1–1 giữa 256 token của các omic có độ nhiễu khác nhau (đặc biệt giảm mạnh trên BRCA xuống `87.41%`).
   - `Linear ELU+1` (`90.13%`) kém ổn định rõ rệt ở chế độ ít mẫu (`COADREAD` tụt xuống `84.63 ± 10.83%`).
3. **Đánh giá Tokenizer `OmicToTokens`:**
   - `tok_dmid512` tăng gấp đôi số tham số (`16.4M → 32.7M`) nhưng **làm giảm -1.36% Avg F1** (đặc biệt overfit trên COADREAD `86.05 ± 6.16%`), xác nhận **`d_mid = 256` là điểm cân bằng tối ưu**, không bị thiếu dung lượng biểu diễn.
   - `tok_layernorm` giúp giảm phương sai trên LUNG (`±0.73%`) và COADREAD (`±1.35%`) nhưng giảm nhẹ điểm trung bình (-0.57%) do Pre-LN trong `TokenCrossAttention` (`norm_q`, `norm_kv`) đã đảm nhiệm chuẩn hóa đầu vào.
