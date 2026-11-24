# Tài liệu cập nhật `main.pdf` theo TokenOSCAR (`256×32`)

Thư mục này chứa toàn bộ hình ảnh mới và danh sách các điểm cần cập nhật khi chuyển báo cáo (`main.pdf`) từ OSCAR `1×128` sang **TokenOSCAR (`256×32`)** theo protocol **matched 5-fold**.

---

## 1. Danh mục hình ảnh trong `material/`

### `material/token_oscar_biomarker/` — Hình Chương 5 (BRCA, TokenOSCAR `256×32`)
- `fig5_1_tokenoscar_consensus_annotated.png` — Thay **Fig 5.1** (5-fold consensus gradient, top-12 mỗi subtype).
- `fig5_2_tokenoscar_cohens_d.png` — Thay **Fig 5.2** (one-versus-rest Cohen's *d* cho 94 consensus features).
- `fig5_3_tokenoscar_cnv_mrna_her2.png` — Thay **Fig 5.3** (đồng thuận CNV vs mRNA trên HER2-enriched).
- `fig5_4_tokenoscar_gate.png` — Thay **Fig 5.4** (trung bình gate và tỉ lệ phần tử `> 0.5` theo subtype trên tập val).

### `material/token_oscar/` — Hình so sánh & khảo sát TokenOSCAR (3 cohort)
- `fig_tokenoscar_vs_baselines.png` — Macro-F1 (mean ± std, 5 fold) của TokenOSCAR `256×32` so với OSCAR `1×128` và 6 baseline.
- `fig_tokenoscar_per_fold.png` — Macro-F1 theo từng fold: TokenOSCAR `256×32` vs OSCAR `1×128`.
- `fig_token_size_sweep.png` — Khảo sát kích thước token `T × M` (`128×32`, `128×64`, `128×128`, `256×32`, `256×64`), kèm số tham số và peak VRAM.

### `material/oscar_1x128/` — Hình gốc của OSCAR `1×128` (lấy từ nhánh `draft`, giữ để đối chiếu)
- `cohens_d_by_subtype.png`, `cnv_mrna_concordance_her2.png`, `gate_by_subtype.png`.

---

## 2. Các nội dung và claim trong `main.pdf` cần thay đổi

Không chỉ đổi số liệu, một số luận điểm chính của thesis hiện tại không còn đúng với TokenOSCAR `256×32` và cần viết lại.

### 2.1. Kiến trúc (Chương 3 & 4) — *Ảnh hưởng lớn nhất*
- **Nội dung cũ:** OSCAR thay softmax attention bằng gate trên vector latent 128 chiều ("no gene softmax", Hình 3.3, Eq. 3.3, mục 3.4) vì chuỗi có độ dài 1 làm softmax suy biến.
- **Cần đổi:** TokenOSCAR biểu diễn mỗi omic thành chuỗi `T = 256` token chiều `M = 32` (`OmicToTokens`) và dùng **multi-head softmax cross-attention thật (`256×256`, 4 heads)** theo thứ tự `CNV → mRNA → Proteomics`, kết hợp `TokenGatedFusion` từng token và self-attention trên `3T + 1 = 769` token với `[CLS]`.
- **Phần bị ảnh hưởng:** Hình 3.1–3.3, mô tả kiến trúc Chương 3, và phần phân tích "representable / non-representable" ở Chương 4.

### 2.2. So sánh với baselines (Bảng 5.1, mục 5.2, mục 5.7, Abstract)
- **Claim cũ:** *"OSCAR is the highest entry in every column"* / *"ranks first among the baselines on all three cohorts"*.
- **Thực tế theo protocol matched (`REPORT.md`):**
  - **BRCA:** TokenOSCAR `89.31 ± 2.17` F1 (`91.00 ± 1.69` Acc) — cao nhất, nhưng gần như hòa MOT (`89.28 ± 3.19`) trong sai số; vượt OSCAR `1×128` (`87.01 ± 3.41`).
  - **LUNG:** `94.70 ± 3.76` F1 — thấp hơn MoXGATE (`96.02`), MOT (`95.87`), OSCAR `1×128` (`95.72`) do fold 1 dao động.
  - **COADREAD:** `91.34 ± 2.55` F1 — ngang MOGONET (`91.37`), vượt MOT (`87.27`) và MoXGATE (`86.09`), thấp hơn OSCAR `1×128` (`93.43`).
  - **Trung bình 3 cohort:** `91.78` Avg F1 — vượt toàn bộ 6 baseline bên ngoài (`+0.97` đến `+6.76`).
- **Cần đổi:** Bỏ claim đứng đầu mọi cột; chuyển sang mô tả hiệu năng **cạnh tranh / đứng đầu trên BRCA** và **vượt toàn bộ 6 baseline bên ngoài theo trung bình 3 cohort**.

### 2.3. Giao thức đánh giá (mục 3.8 & 5.1)
- **Nội dung cũ:** Stratified 5-fold với early stopping chọn best checkpoint trên val.
- **Cần đổi:** Mô tả đúng **matched 5-fold protocol**: outer 5 fold chia sẵn → inner 20% stratified val (`seed=42`) chọn epoch đầu tiên đạt macro-F1 cao nhất (tối đa 120 epoch) → retrain trên 100% tập train của fold đúng số epoch đó → chấm 1 lần duy nhất trên test fold.
- **Lưu ý cho hình biomarker (Fig 5.1–5.4):** Do script đánh giá gốc không lưu checkpoint, các hình biomarker được chạy từ 5 model retrain lại với đúng `chosen_epoch` của báo cáo. Trung bình macro-F1 retrain là `89.40%` (so với `89.31%` trong bảng; từng fold lệch tối đa ~2.3 điểm ở fold 2 do GPU không hoàn toàn deterministic). Cần ghi rõ chi tiết này ở phần phương pháp.

### 2.4. Ablation (Bảng 5.2 & 5.3, mục 5.3–5.4)
- **Hiện trạng:** Bảng 5.2 (bỏ gate, bỏ token self-attn) và Bảng 5.3 (đổi thứ tự cross-attention) hiện vẫn là số của OSCAR `1×128`, chưa chạy lại cho TokenOSCAR.
- **Cần đổi:** Nếu chưa chạy ablation cho TokenOSCAR, **không** dùng các bảng này để kết luận cho TokenOSCAR; chỉ giữ nếu ghi rõ đó là kết quả của phiên bản `1×128`, hoặc thay bằng bảng/hình khảo sát kích thước token `T × M` (`fig_token_size_sweep.png`).

### 2.5. Phân tích Biomarker BRCA (Fig 5.1–5.3, mục 5.5, Abstract)
| Điểm trong `main.pdf` hiện tại | Cập nhật cho TokenOSCAR `256×32` |
|---|---|
| `51` consensus features (`16` LumA, `5` LumB, `9` Basal, `21` HER2) | **`94` consensus features** (`21` LumA, `19` LumB, `21` Basal, `33` HER2) |
| `45/51` pass DE (`16/16`, `5/5`, `4/9`, `20/21`) | **`72/94` pass DE** (`16/21` LumA, `12/19` LumB, `12/21` Basal, `32/33` HER2) |
| 5 gene chung CNV & mRNA trên HER2 | **4 cặp chung CNV & mRNA, đều cùng dấu**: `GRB7`, `PGAP3`, `PNMT` (HER2) và `PNMT` (Basal) |
| Hai nomination `TBC1D31` và `GAS2L3` | **`GAS2L3` không còn xuất hiện** trong consensus; `TBC1D31` vẫn có ở LumA và LumB (mRNA) |
| Basal-like dị biệt, ít pass DE (`4/9`) | Basal-like có `12/21` pass DE và xuất hiện rõ tín hiệu liên quan `17q12` / `ERBB2` (ngược dấu) cùng `INPP4B`, `CKIT` |
| LumA & LumB giàu gene mitotic (Bảng 5.4 Enrichr) | Chưa chạy lại bước 3 (Enrichr) cho danh sách mới của TokenOSCAR; không nên giữ nguyên Bảng 5.4 cũ nếu đã đổi Fig 5.1–5.2 |

### 2.6. Gated Fusion (Fig 5.4, mục 5.6)
- **Nội dung cũ:** Vector gate 128 chiều; mean mRNA `0.52–0.54`, RPPA `~0.50`; tỉ lệ `> 0.5` là `0.56–0.58` (mRNA) và `0.51–0.53` (RPPA).
- **Cập nhật cho TokenOSCAR `256×32`:** Gate là tensor `(256, 32)` từng token; tổng hợp trên toàn bộ `256×32` phần tử mỗi mẫu:
  - **mRNA:** mean `0.50–0.51`, tỉ lệ `> 0.5` khoảng `0.48–0.53` (LumA/Basal `~0.53`, LumB/HER2 `~0.48`).
  - **Proteomics:** mean `0.52–0.53`, tỉ lệ `> 0.5` khoảng `0.64–0.70` (LumA `0.70`, Basal `0.69`, LumB `0.64`, HER2 `0.64`, std giữa các fold `0.02–0.05`).
- **Cần đổi:** Cập nhật các con số trên, đổi "128-dimensional vector" thành token-wise gate `256×32`, và thống nhất tên gọi `Proteomics` thay cho `RPPA`.

---

## 3. Tái lập hình từ code trong `metrix/`

Chạy từ thư mục gốc của repo (cần `PYTHONPATH=.` vì các script nằm trong `metrix/`):

```bash
# 1. Vẽ 3 hình so sánh & khảo sát T×M từ lightning_logs/
.venv/bin/python metrix/plot_token_oscar_report.py

# 2. Retrain 5 fold BRCA của TokenOSCAR 256×32 và lưu weights
PYTHONPATH=. .venv/bin/python metrix/token_oscar_retrain_save.py

# 3. Chạy consensus + DE và vẽ Fig 5.1–5.3
PYTHONPATH=. .venv/bin/python metrix/token_oscar_biomarker_figs.py

# 4. Tính gate và vẽ Fig 5.4
PYTHONPATH=. .venv/bin/python metrix/token_oscar_gate_fig.py
```
