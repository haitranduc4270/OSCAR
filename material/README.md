# Tài liệu cập nhật `main.pdf` theo Proposed Framework (`256×32`)

Thư mục này chứa toàn bộ hình ảnh mới (chú thích tiếng Anh) và danh sách các điểm cần cập nhật cho `main.pdf` theo **Proposed method / Proposed framework** (cấu hình mặc định `T = 256, M = 32`) và protocol **matched 5-fold**.

- **Tên gọi:** không dùng tên "OSCAR" hay "TokenOSCAR" trong báo cáo; gọi thống nhất là **Proposed method** hoặc **Proposed framework**.
- **Baselines:** bỏ **MOT** (không phù hợp để so sánh cùng); chỉ so sánh với 5 baseline: `MoXGATE`, `MoGCN`, `MOGONET`, `moBRCA-net`, `DeepMoIC`.
- **Không đưa phần khảo sát `T×M` sweep** vào báo cáo.

---

## 1. Danh mục hình ảnh trong `material/`

### `material/token_oscar_biomarker/` — Hình Chương 5 (BRCA, Proposed method `256×32`)
- `fig5_1_tokenoscar_consensus_annotated.png` — Thay **Fig 5.1** (5-fold consensus gradient, top-12 mỗi subtype).
- `fig5_2_tokenoscar_cohens_d.png` — Thay **Fig 5.2** (one-versus-rest Cohen's *d* cho 94 consensus features).
- `fig5_3_tokenoscar_cnv_mrna_her2.png` — Thay **Fig 5.3** (đồng thuận CNV vs mRNA trên HER2-enriched).
- `fig5_4_tokenoscar_gate.png` — Thay **Fig 5.4** (trung bình gate và tỉ lệ phần tử `> 0.5` theo subtype trên tập val).

### `material/token_oscar/` — Hình so sánh baseline & theo fold (3 cohort, tiếng Anh)
- `fig_proposed_vs_baselines.png` — Macro-F1 (mean ± std, 5 folds) của **Proposed method** so với 5 baseline (`MoXGATE`, `MoGCN`, `MOGONET`, `moBRCA-net`, `DeepMoIC`).
- `fig_proposed_per_fold.png` — Macro-F1 theo từng fold của **Proposed method** trên BRCA, LUNG, COADREAD.

### `material/oscar_1x128/` — Hình cũ từ nhánh `draft` (giữ để đối chiếu)
- `cohens_d_by_subtype.png`, `cnv_mrna_concordance_her2.png`, `gate_by_subtype.png`.

---

## 2. Các nội dung và claim trong `main.pdf` cần thay đổi

### 2.1. Kiến trúc (Chương 3 & 4) — *Ảnh hưởng lớn nhất*
- **Nội dung cũ:** mô tả cơ chế gate thay softmax trên vector 128 chiều ("no gene softmax", Hình 3.3, Eq. 3.3, mục 3.4).
- **Cần đổi:** mô tả **Proposed framework** với cấu hình mặc định `T = 256, M = 32`: mỗi omic được chiếu thành chuỗi `256` token chiều `32` (`OmicToTokens`), đi qua **multi-head softmax cross-attention (`256×256`, 4 heads)** theo thứ tự `CNV → mRNA → Proteomics`, kết hợp `TokenGatedFusion` từng token và self-attention trên `769` token (`3×256 + [CLS]`).
- **Phần bị ảnh hưởng:** Hình 3.1–3.3, mô tả kiến trúc Chương 3, và phần "representable / non-representable" ở Chương 4.

### 2.2. So sánh với baselines (Bảng 5.1, mục 5.2, mục 5.7, Abstract)
- **Claim cũ:** *"highest entry in every column"* / *"ranks first among the baselines on all three cohorts"*.
- **Thực tế theo protocol matched (5 baseline, không tính MOT):**
  - **BRCA:** Proposed method `89.31 ± 2.17` F1 (`91.00 ± 1.69` Acc) — **đứng đầu**, vượt MoXGATE (`86.82`), moBRCA-net (`83.05`), MoGCN (`81.29`), DeepMoIC (`81.17`), MOGONET (`74.55`).
  - **LUNG:** `94.70 ± 3.76` F1 — thấp hơn MoXGATE (`96.02`) và moBRCA-net (`95.43`) do fold 1 dao động (`87.50`), vượt MoGCN (`94.10`), MOGONET (`93.45`), DeepMoIC (`92.92`).
  - **COADREAD:** `91.34 ± 2.55` F1 — ngang MOGONET (`91.37`), vượt rõ MoXGATE (`86.09`), MoGCN (`84.39`), DeepMoIC (`80.96`), moBRCA-net (`80.60`).
  - **Trung bình 3 cohort:** `91.78` Avg F1 — **vượt toàn bộ 5 baseline** (`+2.14` so với MoXGATE `89.64`, tới `+6.76` so với DeepMoIC `85.02`).
- **Cần đổi:** Bỏ claim đứng đầu mọi cột; nhấn mạnh **đứng đầu rõ rệt trên BRCA**, **cạnh tranh trên LUNG và COADREAD**, và **vượt toàn bộ 5 baseline theo trung bình 3 cohort**.

### 2.3. Giao thức đánh giá (mục 3.8 & 5.1)
- **Cần đổi:** Mô tả đúng **matched 5-fold protocol**: outer 5 fold chia sẵn → inner 20% stratified val (`seed=42`) chọn epoch đầu tiên đạt macro-F1 cao nhất (tối đa 120 epoch) → retrain trên 100% tập train của fold đúng số epoch đó → chấm 1 lần duy nhất trên test fold.
- **Lưu ý cho Fig 5.1–5.4:** được chạy từ 5 model retrain với đúng `chosen_epoch` của báo cáo (mean macro-F1 `89.40%` vs `89.31%` do GPU không hoàn toàn deterministic). Cần ghi rõ chi tiết này ở phần phương pháp.

### 2.4. Ablation (Bảng 5.2 & 5.3, mục 5.3–5.4)
- Hiện vẫn là số của phiên bản cũ, chưa chạy lại cho Proposed framework `256×32`. Tạm thời không gán nhầm cho Proposed framework; cần ghi chú rõ hoặc gỡ bỏ nếu không chạy lại.

### 2.5. Phân tích Biomarker BRCA (Fig 5.1–5.3, mục 5.5, Abstract)
| Điểm trong `main.pdf` hiện tại | Cập nhật cho Proposed framework (`256×32`) |
|---|---|
| `51` consensus features (`16` LumA, `5` LumB, `9` Basal, `21` HER2) | **`94` consensus features** (`21` LumA, `19` LumB, `21` Basal, `33` HER2) |
| `45/51` pass DE (`16/16`, `5/5`, `4/9`, `20/21`) | **`72/94` pass DE** (`16/21` LumA, `12/19` LumB, `12/21` Basal, `32/33` HER2) |
| 5 gene chung CNV & mRNA trên HER2 | **4 cặp chung CNV & mRNA, đều cùng dấu**: `GRB7`, `PGAP3`, `PNMT` (HER2) và `PNMT` (Basal) |
| Hai nomination `TBC1D31` và `GAS2L3` | **`GAS2L3` không còn xuất hiện** trong consensus; `TBC1D31` vẫn có ở LumA và LumB (mRNA) |
| Basal-like dị biệt, ít pass DE (`4/9`) | Basal-like có `12/21` pass DE và xuất hiện rõ tín hiệu liên quan `17q12` / `ERBB2` (ngược dấu) cùng `INPP4B`, `CKIT` |
| Bảng 5.4 (Enrichr) | Chưa chạy lại bước 3 (Enrichr) cho danh sách mới; không giữ nguyên Bảng 5.4 cũ nếu đã đổi Fig 5.1–5.2 |

### 2.6. Gated Fusion (Fig 5.4, mục 5.6)
- **Cập nhật cho Proposed framework (`256×32`):** Gate là tensor `(256, 32)` từng token; tổng hợp trên toàn bộ `256×32` phần tử mỗi mẫu:
  - **mRNA:** mean `0.50–0.51`, tỉ lệ `> 0.5` khoảng `0.48–0.53`.
  - **Proteomics:** mean `0.52–0.53`, tỉ lệ `> 0.5` khoảng `0.64–0.70` (std giữa các fold `0.02–0.05`).
- Đổi "128-dimensional vector" thành token-wise gate `256×32`, và thống nhất tên gọi `Proteomics` thay cho `RPPA`.

---

## 3. Tái lập hình từ code trong `metrix/`

```bash
# 1. Vẽ 2 hình so sánh baseline & theo fold (tiếng Anh)
.venv/bin/python metrix/plot_token_oscar_report.py

# 2. Retrain 5 fold BRCA (256x32) và lưu weights
PYTHONPATH=. .venv/bin/python metrix/token_oscar_retrain_save.py

# 3. Chạy consensus + DE và vẽ Fig 5.1–5.3
PYTHONPATH=. .venv/bin/python metrix/token_oscar_biomarker_figs.py

# 4. Tính gate và vẽ Fig 5.4
PYTHONPATH=. .venv/bin/python metrix/token_oscar_gate_fig.py
```
