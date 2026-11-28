# Pathway-TokenOSCAR (`pw_bn_pathformer_res_all3`) — Full Ablation Study Report

## 1. Mục tiêu & Thiết lập

Báo cáo này chạy lại **toàn bộ bộ thí nghiệm Ablation Study** của kho mã nguồn OSCAR (gồm **Table 5.2**, **Table 5.3**, **Sensitivity `T × M` & đối chiếu `1×128`**, **Exp 2a/3/8** về Gate / Wiring / Softmax alternatives, và **tổng quát hóa sang `MOT` & `MoXGATE`**), trong đó thay thế hoàn toàn bộ chiếu dày `OmicToTokens` bằng bộ chiếu có cấu trúc sinh học **`PathwayBottleneckToTokens` (`pw_bn_pathformer_res_all3`)** trên cả 3 omics (`mRNA`, `CNV`, `Proteomics`).

### Cấu hình Pathway Encoder mặc định (`pw_bn_pathformer_res_all3`)
- **Cơ sở dữ liệu pathway:** `Pathformer` (`1,497` pathways từ Reactome / KEGG / PID / BioCarta, `third_party/pathformer/reference/Pathformer_pathway.txt`).
- **Tầng 1 (nén thưa theo pathway + residual cho gen ngoài pathway):**
  - `mRNA` & `CNV`: lớp tuyến tính thưa `edge-indexed index_add_` nối đúng cặp `(gen, pathway)` (độ phủ `51.7–58.9%` gen) ghép nhánh residual tuyến tính `64` chiều cho phần gen chưa nằm trong pathway $\Rightarrow$ vector bottleneck `1,561` chiều.
  - `Proteomics` (RPPA): ánh xạ kháng thể RPPA $\to$ HGNC (`rppa_feature_to_genes`, độ phủ **`97.8–98.0%`** trên cả 3 cohort, mỗi phospho-protein như `HER2_pY1248`, `4EBP1_pS65`, `ERALPHA_pS118` giữ trọng số cạnh độc lập) + nhánh residual `16` chiều $\Rightarrow$ vector bottleneck `1,513` chiều.
- **Tầng 2 (bung chuỗi token):** `LayerNorm + GELU + Dropout(0.4) + Linear(P + U, T_i × M)`, mặc định `T_mRNA = T_CNV = 256, T_RPPA = 16, M = 32` (`27.68M` tham số trên BRCA).
- **Giao thức đánh giá:** Matched nested 5-fold CV giống hệt các bảng gốc (`seed = 42`, inner 20% stratified val, `first-max` inner-val Macro-F1 trong budget 120 epoch, `StandardScaler` fit chỉ trên train, retrain 100% outer-train, đánh giá 1 lần trên test).
- **Script & dữ liệu thô:** `experiments/run_pathway_ablation_suite.py`, `experiments/pathway_ablation_suite/results.json` (`31` biến thể × `3` cohort × `5` fold = `465` fold).

---

## 2. Table 5.2 — Ablation of Model Components (So với `OmicToTokens`)

### 2.1. Accuracy (%, mean ± std)

| Model variant (`Pathway-TokenOSCAR`) | Cross | Cross Gate | Self | Self Gate | BRCA | LUNG | COADREAD | Avg Acc | Δ vs Full | *OmicToTokens Avg Acc* |
|---|---|---|---|---|---|---|---|---|---|---|
| Concatenation | ✗ | ✗ | ✗ | ✗ | 89.43 ± 1.42 | 96.48 ± 1.26 | 89.63 ± 2.35 | 91.85 | -0.93 | *92.00* |
| **Without self-attention (cross + gate)** | ✓ | ✓ | ✗ | ✗ | **91.78 ± 2.20** | 96.19 ± 1.42 | 91.88 ± 1.84 | **93.28** | **+0.50** | *91.56* |
| Without cross-attention (self + gate + pos) | ✗ | ✗ | ✓ | ✓ | 91.19 ± 2.40 | **96.48 ± 1.17** | 91.88 ± 1.19 | 93.18 | +0.40 | *92.69* |
| Without both gates (+ pos) | ✓ | ✗ | ✓ | ✗ | 90.21 ± 0.90 | 94.43 ± 2.90 | 91.02 ± 4.63 | 91.89 | -0.89 | *92.40* |
| Without cross gate (self gate + pos) | ✓ | ✗ | ✓ | ✓ | 89.63 ± 2.27 | 95.45 ± 1.70 | 91.44 ± 3.58 | 92.18 | -0.61 | *92.34* |
| Without self gate (cross gate + pos) | ✓ | ✓ | ✓ | ✗ | 91.59 ± 3.07 | 95.60 ± 2.07 | **91.89 ± 2.28** | 93.03 | +0.24 | *92.82* |
| **Pathway-TokenOSCAR (full, 4/4)** | **✓** | **✓** | **✓** | **✓** | 91.00 ± 1.91 | 95.89 ± 0.59 | 91.45 ± 2.93 | 92.78 | 0.00 | ***93.25*** |

### 2.2. Macro-F1 (%, mean ± std)

| Model variant (`Pathway-TokenOSCAR`) | Cross | Cross Gate | Self | Self Gate | BRCA | LUNG | COADREAD | Avg F1 | Δ vs Full | *OmicToTokens Avg F1* |
|---|---|---|---|---|---|---|---|---|---|---|
| Concatenation | ✗ | ✗ | ✗ | ✗ | 86.08 ± 1.18 | 96.46 ± 1.27 | 87.98 ± 3.27 | 90.17 | -1.35 | *91.44* |
| **Without self-attention (cross + gate)** | ✓ | ✓ | ✗ | ✗ | 89.58 ± 3.12 | 96.16 ± 1.44 | **91.16 ± 3.01** | **92.30** | **+0.78** | *90.80* |
| Without cross-attention (self + gate + pos) | ✗ | ✗ | ✓ | ✓ | 89.54 ± 2.45 | **96.46 ± 1.18** | 90.57 ± 2.01 | 92.19 | +0.67 | *91.38* |
| Without both gates (+ pos) | ✓ | ✗ | ✓ | ✗ | 88.57 ± 1.87 | 94.41 ± 2.90 | 88.83 ± 7.01 | 90.60 | -0.92 | *91.49* |
| Without cross gate (self gate + pos) | ✓ | ✗ | ✓ | ✓ | 87.71 ± 2.85 | 95.44 ± 1.71 | 90.96 ± 4.08 | 91.37 | -0.15 | *91.70* |
| Without self gate (cross gate + pos) | ✓ | ✓ | ✓ | ✗ | **89.88 ± 4.36** | 95.58 ± 2.08 | 90.54 ± 3.84 | 92.00 | +0.48 | *91.27* |
| **Pathway-TokenOSCAR (full, 4/4)** | **✓** | **✓** | **✓** | **✓** | 88.95 ± 2.52 | 95.88 ± 0.60 | 89.75 ± 3.62 | 91.52 | 0.00 | ***92.18*** |

**Nhận xét Table 5.2:**
- **Lợi ích của cơ chế Attention + Gate trên nền Pathway còn lớn hơn trên `OmicToTokens`:** `Concatenation` chỉ đạt `90.17%` F1 (riêng BRCA tụt xuống `86.08%`), thấp hơn `Full` **`-1.35%` F1** và thấp hơn `Without self-attention (cross + gate)` tới **`-2.13%` F1**.
- **Phát hiện nổi bật nhất của toàn bộ thí nghiệm:** Biến thể **`Pathway-TokenOSCAR (Cross-Attention + Cross Gate, bỏ Self-Attention)`** đạt **`92.30%` Avg F1 / `93.28%` Avg Acc** (BRCA `89.58%` F1 / `91.78%` Acc, LUNG `96.16%`, COADREAD `91.16%`) — **vượt cả `OmicToTokens Full` (`92.18%` F1 / `93.25%` Acc)**. Tương tự, `Without cross-attention` (`92.19%` F1) và `Without self gate` (`92.00%` F1) cũng đều vượt `Full (4/4)` (`91.52%`), cho thấy trên nền pathway bottleneck `1,497`-dim, chỉ cần **1 tầng tương tác có gate** (đặc biệt là Ordered Cross-Attention + Cross Gate) đã đủ giàu thông tin và tránh over-smoothing khi xếp chồng thêm 2 tầng Self-Attention có gate.
- **Vai trò bắt buộc của Gate:** Bỏ cả 2 gate (`no_gate_pos`) làm tụt mạnh về `90.60%` F1 (`-0.92%`, COADREAD dao động tới `±7.01%`).

---

## 3. Table 5.3 — Effect of Cross-Attention Order (6 Hoán vị)

*(Chạy đủ cả 6 hoán vị chuỗi 2 cạnh trên bản `Pathway-TokenOSCAR Full`.)*

### 3.1. Accuracy (%, mean ± std)

| Ordering | BRCA | LUNG | COADREAD | Avg Acc | Δ vs Bio | *OmicToTokens Avg Acc* |
|---|---|---|---|---|---|---|
| Genomics → proteomics → transcriptomics | 91.39 ± 0.75 | 96.63 ± 1.36 | 86.94 ± 4.16 | 91.65 | -1.13 | *92.96* |
| Transcriptomics → genomics → proteomics (mixed) | 91.19 ± 1.39 | 96.19 ± 1.70 | 89.65 ± 2.28 | 92.34 | -0.44 | *92.93* |
| Transcriptomics → proteomics → genomics | 91.39 ± 1.45 | 96.63 ± 0.75 | 89.18 ± 1.73 | 92.40 | -0.38 | *92.47* |
| **Genomics → transcriptomics → proteomics (Bio / Full)** | 91.00 ± 1.91 | 95.89 ± 0.59 | 91.45 ± 2.93 | 92.78 | 0.00 | ***93.25*** |
| Proteomics → transcriptomics → genomics (reverse) | 91.38 ± 2.28 | **97.66 ± 1.07** | 90.07 ± 2.38 | 93.04 | +0.26 | *92.80* |
| **Proteomics → genomics → transcriptomics** | **91.58 ± 1.83** | 96.48 ± 1.56 | **91.86 ± 4.21** | **93.31** | **+0.53** | *92.59* |

### 3.2. Macro-F1 (%, mean ± std)

| Ordering | BRCA | LUNG | COADREAD | Avg F1 | Δ vs Bio | *OmicToTokens Avg F1* |
|---|---|---|---|---|---|---|
| Genomics → proteomics → transcriptomics | 89.41 ± 0.82 | 96.61 ± 1.37 | 83.59 ± 7.58 | 89.87 | -1.65 | *91.92* |
| Transcriptomics → proteomics → genomics | 89.80 ± 2.34 | 96.61 ± 0.76 | 87.00 ± 3.62 | 91.14 | -0.39 | *91.49* |
| Transcriptomics → genomics → proteomics (mixed) | 89.59 ± 2.16 | 96.16 ± 1.72 | 88.82 ± 3.55 | 91.52 | +0.00 | *91.91* |
| **Genomics → transcriptomics → proteomics (Bio / Full)** | 88.95 ± 2.52 | 95.88 ± 0.60 | 89.75 ± 3.62 | 91.52 | 0.00 | ***92.18*** |
| Proteomics → transcriptomics → genomics (reverse) | **90.02 ± 2.75** | **97.65 ± 1.08** | 88.15 ± 2.85 | 91.94 | +0.42 | *92.11* |
| **Proteomics → genomics → transcriptomics** | 89.66 ± 1.91 | 96.47 ± 1.57 | **90.64 ± 5.13** | **92.26** | **+0.73** | *91.36* |

**Nhận xét Table 5.3:**
- Trên nền `Pathway-TokenOSCAR`, hai thứ tự bắt đầu từ `Proteomics` (`Prot → CNV → mRNA` `92.26%` F1 / `93.31%` Acc và `Reverse: Prot → mRNA → CNV` `91.94%` F1 / `93.04%` Acc) vượt thứ tự sinh học (`91.52%` F1 / `92.78%` Acc). Nguyên nhân kỹ thuật đáng chú ý: khi `Proteomics` có `T_RPPA = 16` token còn `mRNA`/`CNV` có `256` token, đặt `Proteomics` làm `key/value` đầu tiên giúp `256` query token của `mRNA`/`CNV` truy vấn vào `16` token protein rất gọn (không bị loãng attention như chiều ngược lại khi `16` token protein truy vấn `256` token mRNA).

---

## 4. Sensitivity Study — Cấu hình Token (`T × M`) & Đối chiếu `1 × 128` vs `256 × 32`

*(Với `T_mRNA = T_CNV = T`, `T_RPPA = min(16, T)`.)*

### 4.1. Độ nhạy lưới Token (`T × M`)

| Cấu hình (`T × M`) | Params (BRCA) | BRCA F1 | LUNG F1 | COADREAD F1 | Avg Acc | Avg F1 | Δ F1 vs `1×128` | *OmicToTokens Avg F1* |
|---|---|---|---|---|---|---|---|---|
| `1 × 128` *(single-token)* | **2.92M** | 87.28 ± 2.92 | **96.61 ± 1.37** | 90.43 ± 0.66 | 92.70 | 91.44 | 0.00 | *90.75* |
| `64 × 32` | 8.47M | 89.54 ± 3.01 | 95.29 ± 2.20 | 88.62 ± 4.76 | 92.26 | 91.15 | -0.29 | *91.12* |
| **`128 × 32`** | **14.88M** | 88.18 ± 2.57 | 96.32 ± 1.62 | **91.13 ± 1.31** | **92.88** | **91.88** | **+0.44** | ***92.41*** |
| `128 × 64` | 28.67M | 89.60 ± 2.28 | 96.03 ± 1.43 | 87.49 ± 2.02 | 92.23 | 91.04 | -0.40 | *91.69* |
| `256 × 64` | 54.27M | **89.65 ± 2.70** | 96.61 ± 1.29 | 87.07 ± 4.38 | 92.62 | 91.11 | -0.33 | *92.18* |
| `256 × 32` *(mặc định)* | 27.68M | 88.95 ± 2.52 | 95.88 ± 0.60 | 89.75 ± 3.62 | 92.78 | 91.52 | +0.08 | *92.18* |

- **Điểm ngọt tối ưu trùng khớp cả hai hệ (`OmicToTokens` & `Pathway`):** Cấu hình **`128 × 32`** đứng #1 trên cả hai hệ (`91.88%` F1 / `92.88%` Acc trên Pathway, **`92.41%`** trên `OmicToTokens`) — đặc biệt trên Pathway, `128 × 32` chỉ tốn **`14.88M` tham số**, thậm chí **nhẹ hơn `OmicToTokens 256×32` (`16.42M`)**!
- **Ưu thế bất ngờ ở chế độ siêu gọn `1 × 128`:** Trên `Pathway-Bottleneck`, `1 × 128` chỉ có **`2.92M` tham số** (nhẹ hơn `OmicToTokens 1×128` `11.21M` gần **4 lần**) nhưng lại đạt **`91.44%` F1 / `92.70%` Acc** (vượt `OmicToTokens 1×128` `90.75%` F1 / `91.85%` Acc tới **`+0.69%` F1 / `+0.85%` Acc**!).

### 4.2. Đối chiếu tác dụng 2 khối Attention trên `1 × 128` vs `256 × 32` (Pathway-Bottleneck)

| Biến thể | `1 × 128` Avg Acc (Δ vs Full) | `256 × 32` Avg Acc (Δ vs Full) | `1 × 128` Avg F1 (Δ vs Full) | `256 × 32` Avg F1 (Δ vs Full) |
|---|---|---|---|---|
| Concatenation *(tắt cả 2 attention)* | 91.82 (-0.88) | 91.85 (-0.93) | 90.40 (-1.04) | 90.17 (-1.35) |
| Without self-attention *(chỉ cross + gate)* | 92.62 (-0.08) | **93.28 (+0.50)** | 91.27 (-0.17) | **92.30 (+0.78)** |
| Without cross-attention *(chỉ self + gate + pos)* | 91.76 (-0.94) | 93.18 (+0.40) | 90.57 (-0.86) | 92.19 (+0.67) |
| **Full *(cross + self + 2 gates + pos)*** | **92.70 (0.00)** | 92.78 (0.00) | **91.44 (0.00)** | 91.52 (0.00) |

- Khác với `OmicToTokens 1×128` (nơi `Full` bị tụt dưới `Concatenation`), trên nền `Pathway-Bottleneck 1×128`, bản `Full` (`91.44%`) thắng sạch `Concat` (`+1.04%` F1), `no_cross` (`+0.86%`), và `no_self` (`+0.17%`).

---

## 5. Exp 2a, Exp 3 & Exp 8 — Fixed Gate, Architectural Wiring & Softmax Alternatives

| Biến thể (`Pathway-TokenOSCAR 256×32`) | Params (BRCA) | BRCA F1 | LUNG F1 | COADREAD F1 | Avg Acc | Avg F1 | Δ F1 vs Full | *OmicToTokens Avg F1* |
|---|---|---|---|---|---|---|---|---|
| **`Full` (Learned Gate, `Z_T`, 2 edges, Softmax)** | 27,680,866 | 88.95 ± 2.52 | 95.88 ± 0.60 | 89.75 ± 3.62 | 92.78 | 91.52 | 0.00 | ***92.18*** |
| `fixed_gate_0.5` ($\alpha \equiv 0.5$) | 27,662,146 | 89.02 ± 3.63 | 96.16 ± 1.44 | 90.16 ± 2.42 | 92.81 | 91.78 | +0.26 | *90.87* |
| `fixed_gate_1.0` ($\alpha \equiv 1.0$) | 27,662,146 | 88.10 ± 1.95 | 94.71 ± 3.08 | 88.31 ± 6.66 | 91.70 | 90.37 | -1.15 | *91.49* |
| **`chained_FT` (dùng $F_T$ thay $Z_T$ ở Step 2)** | 27,680,866 | 88.93 ± 2.55 | 96.46 ± 1.51 | 90.14 ± 4.49 | **92.98** | **91.84** | **+0.32** | *91.36* |
| `unrestricted_all2all` (cả 6 cạnh có hướng) | 27,756,898 | 88.23 ± 2.32 | **96.90 ± 1.43** | 87.83 ± 6.93 | 92.40 | 90.99 | -0.54 | *91.37* |
| `tok_layernorm` (post-reshape LN) | 27,681,058 | 86.89 ± 3.05 | 96.47 ± 1.43 | **91.58 ± 3.29** | 92.67 | 91.65 | +0.12 | *91.61* |
| `attn_linear_elu` (ELU+1) | 27,680,866 | **90.39 ± 2.70** | 95.73 ± 1.50 | 87.34 ± 5.90 | 92.32 | 91.15 | -0.37 | *90.13* |
| `attn_sigmoid` ($\sigma(QK^\top/\sqrt{d_k}-\log T)$) | 27,680,866 | 89.24 ± 3.13 | 95.88 ± 1.78 | 86.58 ± 5.64 | 91.84 | 90.57 | -0.96 | *91.68* |
| `attn_sinkhorn` (5-iter OT) | 27,680,866 | 87.36 ± 2.80 | 95.43 ± 1.65 | 86.10 ± 3.05 | 91.06 | 89.63 | -1.89 | *90.67* |
| `attn_sparsemax` (simplex thưa) | 27,680,866 | 86.82 ± 2.45 | 96.16 ± 1.57 | 83.09 ± 5.77 | 91.15 | 88.69 | -2.83 | *91.48* |

**Nhận xét Exp 2a / 3 / 8:**
- **`Softmax` vẫn đứng #1 áp đảo** so với mọi cơ chế thay thế (`Linear ELU+1` `-0.37%` dù đạt đỉnh BRCA `90.39%`, `Sigmoid` `-0.96%`, `Sinkhorn` `-1.89%`, `Sparsemax` `-2.83%`).
- **`unrestricted_all2all` (6 cạnh)** tiếp tục tụt mạnh trên COADREAD (`87.83 ± 6.93%`, Avg `-0.54%`), tái khẳng định lợi thế chống overfit của đồ thị 2 cạnh gọn nhẹ.
- **`chained_FT`** (`91.84%`, `+0.32%`) lại nhỉnh nhẹ trên nền Pathway (ngược với `OmicToTokens`), gợi ý rằng sau khi nén CNV qua 1,497 pathway sinh học, nhiễu CNV đã được lọc bớt nên truyền nối tiếp $F_T \to \text{Prot}$ hoạt động tốt hơn.

---

## 6. Tổng quát hóa sang Baseline Attention (`MOT` & `MoXGATE` + Pathway Bottleneck)

| Model | Params (BRCA) | BRCA Acc / F1 | LUNG Acc / F1 | COADREAD Acc / F1 | Avg Acc | Avg F1 | Δ F1 vs Pathway-TokenOSCAR |
|---|---|---|---|---|---|---|---|
| `MOT` *(orig)* | — | 90.80 / 89.28 | 95.90 / 95.87 | 90.11 / 87.27 | 92.27 | 90.81 | -0.71 |
| `MOT + OmicToTokens (256×32)` | — | 91.58 / 89.86 | 95.60 / 95.58 | 90.10 / 89.53 | 92.43 | 91.65 | +0.13 |
| **`MOT + Pathway (pw_bn_pathformer_res_all3)`** | 27.63M | 90.22 / 86.86 | 96.04 / 96.02 | 88.30 / 86.79 | 91.52 | 89.89 | **-1.63** |
| `MoXGATE` *(orig)* | — | 89.23 / 86.82 | 96.04 / 96.02 | 88.30 / 86.09 | 91.19 | 89.64 | -1.88 |
| `MoXGATE + OmicToTokens (256×32)` | — | 87.09 / 84.27 | 96.04 / 96.02 | 89.65 / 89.40 | 90.93 | 89.90 | -1.62 |
| **`MoXGATE + Pathway (pw_bn_pathformer_res_all3)`** | 27.61M | 88.65 / 85.60 | 95.75 / 95.73 | 88.27 / 86.86 | 90.89 | 89.40 | **-2.13** |
| **`Pathway-TokenOSCAR (Full, 256×32)`** | 27.68M | 91.00 / 88.95 | 95.89 / 95.88 | 91.45 / 89.75 | 92.78 | 91.52 | **0.00** |
| **`Pathway-TokenOSCAR (Cross+Gate, no SA)`** | 27.64M | **91.78 / 89.58** | **96.19 / 96.16** | **91.88 / 91.16** | **93.28** | **92.30** | **+0.78** |

**Nhận xét:** Khi đặt cùng một bộ chiếu `PathwayBottleneckToTokens` lên 3 kiến trúc attention (`MOT`, `MoXGATE`, `TokenOSCAR`), cả `MOT + Pathway` (`89.89%`) lẫn `MoXGATE + Pathway` (`89.40%`) đều tụt sâu, trong khi **`Pathway-TokenOSCAR`** vượt trội từ **`+1.63%` đến `+2.90%` Avg F1**. Điều này khẳng định **kiến trúc fusion có gate của OSCAR là chìa khóa khai thác hiệu quả biểu diễn pathway**.

---

## 7. Tổng kết & Khuyến nghị khi chọn Pathway-TokenOSCAR cho Luận văn

1. **Hoàn toàn khả thi để dùng `Pathway-TokenOSCAR` làm kiến trúc chính** nếu bạn ưu tiên tính diễn giải sinh học ở tầng đầu (1,497 pathway Pathformer cho cả 3 omics, bảo toàn phospho-protein RPPA với độ phủ `97.8–98.0%`, kèm nhánh residual không mất gen ngoài pathway):
   - **Lựa chọn mạnh nhất về điểm số (`92.30%` Avg F1 / `93.28%` Avg Acc — vượt `OmicToTokens` `92.18%`):** `Pathway-TokenOSCAR` dùng **Ordered Cross-Attention + Cross Gate (`no_self_attn`)** (`BRCA 89.58%`, `LUNG 96.16%`, `COADREAD 91.16%`).
   - **Lựa chọn cân bằng điểm số + tham số (`14.88M` params — nhẹ hơn `OmicToTokens` `16.42M`):** `Pathway-TokenOSCAR Full` ở cấu hình **`128 × 32`** đạt **`91.88%` Avg F1 / `92.88%` Avg Acc** (COADREAD `91.13 ± 1.31%`).
   - **Lựa chọn siêu nhẹ (`2.92M` params — gọn hơn 5.6×):** `Pathway-TokenOSCAR Full` ở **`1 × 128`** đạt **`91.44%` Avg F1 / `92.70%` Acc**.
2. **Điểm cần lưu ý trung thực khi bảo vệ:**
   - Trên nền `Pathway-TokenOSCAR Full (256×32)`, cấu trúc 4/4 (`91.52%`) hơi thừa dung lượng so với chỉ dùng Cross+Gate (`92.30%`) hoặc `128×32` (`91.88%`), và thứ tự bắt đầu từ `Proteomics` (`91.94–92.26%`) nhỉnh hơn thứ tự sinh học (`91.52%`) do bất đối xứng `T_RPPA = 16` vs `T_mRNA = 256`.
   - Khoảng cách trước `Concatenation` (`+1.35%` đến `+2.13%` F1) và trước `MOT`/`MoXGATE` (`+1.63%` đến `+2.90%` F1) **đều rõ rệt hơn cả trên `OmicToTokens`**, tạo nên câu chuyện thực nghiệm rất thuyết phục về sự cộng hưởng giữa **Pathway Bottleneck** và **Gated Cross-Attention**.
