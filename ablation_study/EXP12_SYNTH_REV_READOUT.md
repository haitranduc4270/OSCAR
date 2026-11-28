# Exp 12 — Kiểm chứng Readout Đối xứng trên Synthetic Reverse Control (`S1_control_rev_2edge`)

## 1. Mục tiêu & Giả thuyết kiểm chứng

Trong Exp 9 (`experiments/run_exp9_learned_order.py`), thí nghiệm đối chứng dương trên dữ liệu tổng hợp chuỗi XOR (`CNV → mRNA → Proteomics`) ghi nhận:

- `S1_control_bio_2edge` (`CNV → mRNA → Proteomics`, readout chỉ tại nút `Proteomics`): **`100.00 ± 0.00%`** Macro-F1.
- `S1_control_rev_2edge` (`Proteomics → mRNA → CNV`, readout chỉ tại nút `Proteomics`): **`24.10 ± 3.22%`** Macro-F1 (mức ngẫu nhiên 4 lớp).

Một câu hỏi phản biện quan trọng đặt ra: *Liệu mức F1 `24.10%` ở chiều ngược là do bản thân cơ chế truyền tin/học cạnh không hoạt động ở chiều ngược, hay hoàn toàn do thiết kế **readout chỉ đọc nút cuối `Proteomics`** (nút có bậc vào bằng 0 dưới đồ thị ngược `Prot → mRNA → CNV`)?*

Thí nghiệm này kiểm chứng trực tiếp bằng cách chạy biến thể **readout đối xứng** (nối trung bình biểu diễn của cả 3 nút `[CNV, mRNA, Proteomics]`) và **readout tại nút cuối của chiều ngược (`CNV`)** trên đúng 5 seed của dữ liệu tổng hợp Exp 9.

---

## 2. Thiết lập thực nghiệm

- **Dữ liệu:** Đúng bộ dữ liệu tổng hợp XOR chain `generate_synthetic_xor_chain` (`N=800`, 4 lớp, 3 omic `CNV/mRNA/Prot`, 5 seeds `42, 123, 2024, 7, 99`, `max_epochs = 70`).
- **Kiến trúc:** `TokenOSCARModel` (`num_tokens=16, token_dim=32, d_mid=64, use_self_attention=False, learnable_edges=False`), chỉ thay đổi cấu hình `readout_omic` và `edge_cascade_steps`.
- **Script & dữ liệu thô:**
  - Script: `experiments/run_exp12_synth_rev_readout.py`
  - Kết quả thô: `experiments/defense_followup/exp12_synth_rev_readout/synth_rev_readout_results.json`

---

## 3. Kết quả thực nghiệm (5 Seeds)

| Biến thể Synthetic (5 seeds) | Cạnh cố định | Cascade steps | Readout | Test Macro-F1 | Test Acc |
|---|---|---|---|---|---|
| `S1_control_rev_2edge_prot_readout` *(Exp 9 gốc)* | `Prot → mRNA → CNV` | 2-step | Chỉ nút `Proteomics` | 24.10 ± 3.22% | 24.38 ± 3.09% |
| **`S1_control_rev_2edge_sym_readout`** | `Prot → mRNA → CNV` | 2-step | **Đối xứng `[G, T, P]`** | **100.00 ± 0.00%** | **100.00 ± 0.00%** |
| **`S1_control_rev_2edge_1step_sym`** | `Prot → mRNA → CNV` | 1-step | **Đối xứng `[G, T, P]`** | **100.00 ± 0.00%** | **100.00 ± 0.00%** |
| **`S1_control_rev_2edge_cnv_readout`** | `Prot → mRNA → CNV` | 2-step | Nút cuối chiều ngược `CNV` | **100.00 ± 0.00%** | **100.00 ± 0.00%** |
| `S1_control_bio_2edge_sym_readout` | `CNV → mRNA → Prot` | 2-step | Đối xứng `[G, T, P]` | **100.00 ± 0.00%** | **100.00 ± 0.00%** |

---

## 4. Phân tích & Kết luận

1. **Xác nhận hoàn toàn nguyên nhân của `24.10%` ở Exp 9:**
   - Tái lập chính xác `24.10 ± 3.22%` khi giữ `readout_omic='Proteomics'` trên chuỗi ngược `Prot → mRNA → CNV`.
   - Khi thay bằng **readout đối xứng `[G, T, P]`** (cả 2-step lẫn 1-step) hoặc đọc tại nút đích tương ứng của chuỗi ngược (`CNV`), mô hình đạt ngay **`100.00 ± 0.00%` Macro-F1 trên cả 5/5 seed**.
2. **Ý nghĩa phương pháp luận cho báo cáo phản biện:**
   - Cơ chế cross-attention hoàn toàn hoạt động đối xứng theo hướng chuỗi khi đầu ra được gộp đối xứng từ cả 3 omic; hiện tượng sụp về `24.10%` ở `S1_control_rev_2edge` chỉ xảy ra do ràng buộc đọc riêng nút `Proteomics` (kiểm chứng tính truyền tin có hướng vào nút đích cố định `P`), **không** phải do cơ chế học cạnh không vận hành được ở chiều ngược.
   - Phát hiện này cũng hoàn toàn nhất quán với kết quả `S2_symmetric_readout_budget2` trong Exp 9 (khi readout đối xứng, mô hình khôi phục đúng **khung vô hướng (skeleton)** `{CNV–mRNA, mRNA–Prot}` 5/5 seed nhưng có tính đối xứng về chiều mũi tên) và với kết quả trên TCGA (nơi `Biological` và `Reverse` đạt hiệu năng rất gần nhau khi cùng dùng tầng fusion đối xứng).
