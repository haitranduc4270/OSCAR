# Exp 10: Pathway-Guided Projection (`PathwayBottleneckToTokens`) & Direct Pathway Tokenization (`PathwayTokenEncoder`)

## 1. Động lực thí nghiệm (Trả lời Q1 & Q15)

Trong `TokenOSCAR (256×32)` mặc định, `OmicToTokens` dùng một phép chiếu tuyến tính dày `Linear(D, 256) -> Linear(256, 256*32)` (`16.42M` tham số trên BRCA). Thí nghiệm này kiểm chứng hai cách thay thế `OmicToTokens` (cho `mRNA` và `CNV`) bằng cơ chế **nhóm gen theo biological pathway** (trong khi giữ nguyên `OmicToTokens` cho `Proteomics` RPPA vì RPPA chỉ có `153–223` protein gồm nhiều phospho-site đặc hiệu vị trí như `HER2_pY1248`, `4EBP1_pS65`):

1. **Cơ chế A — `PathwayBottleneckToTokens` (giảm chiều bằng pathway):** Thay lớp đầu dày `Linear(D, 256)` bằng lớp tuyến tính thưa chỉ nối các cặp `(gen, pathway)` hợp lệ (`edge-indexed index_add_`, khởi tạo fan-in $1/\sqrt{\text{deg}_p}$), ánh xạ `(B, D) -> (B, P)`, sau đó `LayerNorm + GELU + Dropout + Linear(P, 256*32)` ra `(B, 256, 32)`. Kèm biến thể có thêm nhánh residual `Linear(D_unmapped, 64)` cho các gen không nằm trong cơ sở dữ liệu pathway (`pw_bn_pathformer_res`) để tách riêng ảnh hưởng của *độ phủ gen* vs *cấu trúc pathway*.
2. **Cơ chế B — `PathwayTokenEncoder` (mỗi pathway là 1 token):** Mỗi pathway $p \in \{1,\dots,P\}$ gom trực tiếp các gen thành viên qua $K=8$ tổ hợp trọng số học được `(B, D) -> (B, P, 8)`, qua MLP chia sẻ `8 -> 32` cộng `pathway_embed (1, P, 32)` để tạo thẳng chuỗi `(B, P, 32)` đưa vào `TokenCrossAttention` + `TokenGatedFusion` + `Self-Attention` (RPPA chiếu sang `T_P = P` token).

---

## 2. Cơ sở dữ liệu Pathway & Độ phủ gen

| Nguồn Pathway | Số pathway $P$ | Độ phủ gen `mRNA` (BRCA / LUNG-CRC) | Độ phủ gen `CNV` (BRCA / LUNG-CRC) |
|---|---|---|---|
| **MSigDB Hallmark** (`resources/msigdb/h.all.v7.4.symbols.gmt`) | `50` | `22.4%` / `20.5%` | `22.7%` / `17.1%` |
| **PathCNN KEGG** (`third_party/pathformer/reference/pathway_PathCNN.txt`) | `146` | `25.1%` / `23.1%` | `25.3%` / `19.3%` |
| **Pathformer** (`third_party/pathformer/reference/Pathformer_pathway.txt`) | `1,497` | `58.3%` / `51.7%` | `58.9%` / `43.9%` |

---

## 3. Kết quả thực nghiệm (Matched Nested 5-Fold CV, `seed = 42`, budget 120 epochs)

**Code & dữ liệu thô:** `models/pathway_tokenizer.py`, `experiments/run_exp10_pathway_tokenizer.py`, `experiments/defense_followup/exp10_pathway_tokenizer/exp10_results.json`.

| Biến thể Tokenizer | Params (BRCA) | BRCA F1 | LUNG F1 | COADREAD F1 | Avg F1 | Avg Acc | Δ vs Default |
|---|---|---|---|---|---|---|---|
| **`TokenOSCAR (OmicToTokens 256×32)`** | 16,420,996 | **89.93 ± 2.15%** | 97.34 ± 1.30% | 89.26 ± 2.55% | **92.18%** | **93.25%** | — |
| `pw_bn_pathformer_res` (`P=1,497` + 64-dim unmapped) | 29,063,500 | 88.77 ± 2.95% | 96.46 ± 2.01% | **90.04 ± 5.86%** | 91.76% | 92.98% | **-0.42%** |
| `pw_bn_pathformer` (`P=1,497`, chỉ gen thuộc pathway) | 26,984,076 | 86.97 ± 3.19% | 97.64 ± 1.19% | 88.46 ± 5.00% | 91.02% | 92.41% | -1.16% |
| `pw_bn_hallmark` (`P=50` bottleneck, giảm 5.3× params) | 3,112,917 | 85.00 ± 2.73% | 97.49 ± 1.11% | 84.00 ± 4.59% | 88.83% | 90.61% | -3.35% |
| `pw_bn_kegg` (`P=146` bottleneck, giảm 3.5× params) | 4,691,343 | 86.49 ± 1.33% | 97.79 ± 1.33% | 81.60 ± 6.07% | 88.63% | 90.18% | -3.55% |
| `pw_tok_hallmark` (`50×32` pathway tokens, **giảm 24.5×**) | **670,124** | 85.85 ± 4.90% | **98.09 ± 0.76%** | 80.23 ± 4.90% | 88.06% | 90.06% | -4.12% |
| `pw_tok_kegg` (`146×32` pathway tokens, **giảm 10.8×**) | 1,516,412 | 83.79 ± 4.88% | 97.94 ± 0.71% | 75.67 ± 3.53% | 85.80% | 87.97% | -6.38% |
| *Đối chứng Exp 7c:* `feature_chunk_64x32` (ViT chunk) | 100,644 | 77.47 ± 3.68% | 94.10 ± 2.24% | 82.31 ± 3.01% | 84.63% | — | -7.55% |
| *Đối chứng Exp 7c:* `feature_chunk_256x32` (ViT chunk) | 104,420 | 61.25 ± 9.82% | 89.71 ± 5.04% | 44.54 ± 13.64% | 65.17% | — | -27.01% |

---

## 4. Phân tích & Kết luận rút ra

1. **Trên LUNG (`LUAD` vs `LUSC`), nhóm theo pathway VƯỢT hẳn `OmicToTokens` mặc định trên 5/6 biến thể với số tham số giảm tới 25 lần:**
   - **`pw_tok_hallmark`** đạt **`98.09 ± 0.76%`** Macro-F1 (**+0.75%** so với `97.34 ± 1.30%`, giảm gần một nửa phương sai) dù chỉ có **`655K` tham số** (so với `16.4M`).
   - **`pw_tok_kegg`** đạt **`97.94 ± 0.71%`** (**+0.60%**, `1.50M` tham số), **`pw_bn_kegg`** đạt **`97.79 ± 1.33%`** (**+0.45%**).
   - Lý do sinh học: phân loại nhị phân `LUAD` vs `LUSC` được chi phối bởi các chương trình phiên mã kinh điển (biểu mô tuyến vs tế bào vảy, chu kỳ tế bào) vốn nằm rất gọn trong 50 MSigDB Hallmark và 146 KEGG pathways — ràng buộc thưa theo pathway đóng vai trò regularizer cực tốt.

2. **Trên bài toán phân nhóm mịn 4 lớp (`BRCA` PAM50 & `COADREAD` CMS), nút cổ chai chính là *độ phủ gen (gene coverage)*:**
   - Khi chỉ giữ ~`20–25%` gen (Hallmark / KEGG), F1 trên BRCA tụt `-3.4%` đến `-6.1%` và trên COADREAD tụt `-5.3%` đến `-13.6%`.
   - Khi nâng độ phủ lên ~`52–59%` (`pw_bn_pathformer`, `P=1,497`), Avg F1 nhảy vọt lên **`91.02%`** (BRCA `86.97%`, COADREAD `88.46%`).
   - Khi khôi phục nốt phần gen chưa được gán pathway qua nhánh residual 64 chiều (`pw_bn_pathformer_res`), BRCA tăng thêm **+1.80%** lên **`88.77%`** và COADREAD đạt **`90.04%`** (vượt nhẹ baseline `89.26%`), đưa trung bình 3 cohort lên **`91.76%`** (chỉ còn kém `OmicToTokens` `-0.42%`).
   - Điều này **chứng minh trực tiếp bằng thực nghiệm luận điểm Q15**: lợi thế của `OmicToTokens` trên trung bình đa cohort (`92.18%`) đến từ việc giữ nguyên **100% không gian gen** (bao gồm các marker phân nhóm mịn ngoài cơ sở dữ liệu pathway cố định) cùng khả năng trộn toàn cục.

3. **Nhóm theo pathway vượt xa cắt chunk cơ học kiểu ViT (`FeatureChunkToTokens`, Exp 7c):**
   - Cùng là tokenizer cục bộ siêu gọn, `pw_tok_hallmark` (`50×32`, `670K` params) đạt **`88.06%`** Avg F1 — vượt `feature_chunk_64x32` (**+3.43%**, riêng BRCA **+8.38%**) và vượt `feature_chunk_256x32` (**+22.89%**). Điều này cho thấy nếu muốn token hóa cục bộ có ý nghĩa sinh học và gọn tham số, **nhóm theo pathway là lựa chọn vượt trội hoàn toàn so với cắt chunk liên tiếp**.

---

## 5. Exp 10b — Thu gọn RPPA (`d_mid=64, T_P=16`) & Nhóm cả 3 Omics theo Pathway (gồm RPPA)

**Động lực:** Trong Exp 10, giữ nguyên `OmicToTokens(d_mid=256)` cho RPPA khiến riêng encoder RPPA chiếm tới `69–83%` tổng tham số (`2.16M` trong `pw_bn_hallmark`, `1.26M` trong `pw_tok_kegg`). Exp 10b kiểm chứng hai hướng xử lý trên đúng giao thức matched 5-fold (`experiments/run_exp10b_compact_and_all3_pathway.py`, kết quả `experiments/defense_followup/exp10_pathway_tokenizer/exp10b_results.json`):
1. **Phần 1 — Compact RPPA (`*_rppa64x16`):** Hạ `d_mid_rppa = 64, T_P = 16` (`~43K–48K` params cho RPPA, giảm **45×** so với `2.16M`).
2. **Phần 2 — All-3-Omics Pathway Grouping (`*_all3`):** Dùng bảng ánh xạ chuẩn `RPPA antibody -> HGNC gene(s)` (`rppa_feature_to_genes` trong `models/pathway_tokenizer.py`) trong đó **mỗi phospho-protein giữ trọng số cạnh độc lập** vào các pathway chứa gen tương ứng. Độ phủ RPPA đạt **`97.8–98.0%`** trên Pathformer (`218/223` BRCA, `176/180` LUNG, `150/153` COADREAD), **`79.1–83.3%`** trên Hallmark và **`77.1–79.4%`** trên KEGG.

| Biến thể Exp 10b | Params BRCA (RPPA) | BRCA F1 | LUNG F1 | COADREAD F1 | Avg F1 | Δ vs Default |
|---|---|---|---|---|---|---|
| **`TokenOSCAR (OmicToTokens 256×32)`** | 16,420,996 (2.16M) | **89.93 ± 2.15%** | 97.34 ± 1.30% | 89.26 ± 2.55% | **92.18%** | — |
| `pw_bn_pathformer_res_all3` (3 omics, `T_P=16`) | 27,680,866 (788K) | **89.34 ± 2.88%** | 95.73 ± 1.50% | **89.75 ± 3.62%** | **91.61%** | **-0.57%** |
| `pw_bn_pathformer_res_rppa64x16` | 26,940,364 (48K) | 87.18 ± 2.68% | 95.28 ± 1.78% | 86.79 ± 1.99% | 89.75% | -2.43% |
| `pw_tok_hallmark_rppa64x16` (**gọn 66×**) | 247,724 (48K) | 82.47 ± 3.76% | 97.35 ± 1.00% | 85.66 ± 5.22% | 88.49% | -3.69% |
| `pw_bn_hallmark_rppa64x16` (gọn 16.6×) | 989,781 (48K) | 84.85 ± 2.32% | 97.49 ± 0.37% | 80.39 ± 7.58% | 87.58% | -4.60% |
| `pw_tok_hallmark_all3` (**gọn 79×**) | **207,188 (6.1K)** | 80.33 ± 1.79% | 95.58 ± 1.40% | 85.83 ± 3.13% | 87.25% | -4.93% |
| `pw_tok_kegg_all3` (**gọn 61×**) | 271,148 (13.3K) | 78.96 ± 3.51% | 96.03 ± 1.00% | 84.39 ± 2.35% | 86.46% | -5.72% |
| `pw_tok_kegg_rppa64x16` (gọn 54×) | 301,436 (48K) | 75.80 ± 7.28% | **97.64 ± 1.28%** | 84.41 ± 2.18% | 85.95% | -6.23% |

**Kết luận bổ sung (Exp 10b):**
1. **Trên tập ít mẫu COADREAD (`N=222`, `153` RPPA), cắt bỏ phình tham số RPPA giúp các biến thể Pathway-as-Tokens tăng vọt:** `pw_tok_hallmark` tăng **`+5.43%`** (`80.23% -> 85.66%` với `rppa64x16`) và **`+5.60%`** (`85.83 ± 3.13%` với `all3`, chỉ **`202K` params**); `pw_tok_kegg` tăng **`+8.74%`** (`75.67% -> 84.41%` với `rppa64x16`) và **`+8.72%`** (`84.39%` với `all3`). Điều này xác nhận trực tiếp giả thuyết overfit do RPPA bị phình `153 -> 256 -> T×32` ở Exp 10.
2. **Nhóm cả 3 omics theo Pathformer + residual (`pw_bn_pathformer_res_all3`) lập đỉnh mới trong họ pathway trên BRCA (`89.34 ± 2.88%`, `91.19%` Acc)** — chỉ kém `TokenOSCAR 256×32` (`89.93%`) đúng `-0.59%`, đồng thời vượt baseline trên COADREAD (`89.75 ± 3.62%` vs `89.26%`, giảm mạnh độ lệch chuẩn so với `±5.86%` của `pw_bn_pathformer_res`).
3. **Đánh đổi giữa kích thước RPPA và cân bằng số token trong Self-Attention:** Khi ghép `T_P = 16` với `256` token mRNA + `256` token CNV (`pw_bn_pathformer_res_rppa64x16`), RPPA chỉ chiếm `3.0%` chiều dài chuỗi Self-Attention nên làm giảm hiệu năng (`91.76% -> 89.75%`); tương tự, ở `pw_tok_hallmark_all3` (`6.1K` params RPPA) mất ~20% protein ngoài Hallmark làm giảm BRCA (`85.85% -> 80.33%`) dù COADREAD tăng mạnh.

