# `bio_material/` — BRCA Biomarker, Gate & Enrichr (Full Architecture, `256×32`)

Thư mục này gom toàn bộ hình ảnh, số liệu và helper modules để tái lập phân tích biomarker trên **BRCA** cho kiến trúc **TokenOSCAR Full hiện tại** (`T = 256, M = 32`, cross gate + self gate + learned `pos_embed`, mean Macro-F1 **89.93%**).

---

## 1. Cấu trúc thư mục

- `figures_v2/` — 4 hình Chương 5 (dùng trực tiếp cho báo cáo):
  - `fig5_1_consensus_annotated_v2.png` — **Fig 5.1**: 5-fold consensus input-gradient features (top-12 mỗi subtype).
  - `fig5_2_cohens_d_v2.png` — **Fig 5.2**: One-vs-rest Cohen's *d* cho toàn bộ 109 consensus features.
  - `fig5_3_cnv_mrna_her2_v2.png` — **Fig 5.3**: Đồng thuận CNV vs mRNA trên HER2-enriched.
  - `fig5_4_gate_v2.png` — **Fig 5.4**: Token-wise gated fusion trên 5 validation fold.
- `results_v2/` — Toàn bộ CSV, báo cáo trung gian và report gốc:
  - `sweep_v2_t256_m32_report.json` — Report gốc của model Full (`89.93%` BRCA Macro-F1).
  - `token_oscar_step1_consensus/` — Danh sách consensus features (`consensus_features.csv`, `consensus_summary.csv`).
  - `token_oscar_step2_de/` — Kết quả Differential Expression (`consensus_with_de.csv`, `consensus_de_pass.csv`, `REPORT_step2_de.md`).
  - `token_oscar_step3_enrichment/` — Kết quả Enrichr (`enrichment_significant_adjP0.05.csv`, `enrichment_all.csv`, `REPORT_step3_enrichment.md`).
  - `token_oscar_report_assets/` — Bảng tổng hợp gate (`token_gate_summary.csv`, `token_gate_samples.csv`) và CNV–mRNA concordance (`cnv_mrna_concordance.csv`).
- `modules/` — Các helper module cần cho pipeline biomarker (`train.py`, `datamodule.py`, `visualize_attention.py`), dùng qua `PYTHONPATH=.:bio_material/modules`.

*(Weights 5 fold BRCA lưu tại `lightning_logs/token_oscar_biomarker/fold_{1..5}.pt`, không commit vào git do dung lượng ~320 MB.)*

---

## 2. Tóm tắt số liệu mới nhất (Full `256×32`, v2)

### Retrain 5-fold BRCA (khớp tuyệt đối với `sweep_v2_t256_m32_report.json`)
- **Chosen epochs:** `[42, 57, 28, 24, 27]`
- **Macro-F1 từng fold (%):** `89.40`, `90.98`, `93.55`, `87.75`, `87.95` → **Mean: 89.93%**

### Consensus & Differential Expression (`q < 0.05` và `|d| ≥ 0.3`)
- **Tổng:** `61 / 109` features pass DE.
- **Theo subtype:** LumA `13/24`, LumB `7/19`, Basal `19/40`, HER2 `22/26`.

### Token-wise Gated Fusion (Fig 5.4, mean across 5 folds)
| Subtype | mRNA mean gate | mRNA frac `> 0.5` | Proteomics mean gate | Proteomics frac `> 0.5` |
|---|---:|---:|---:|---:|
| LumA | 0.502 | 0.507 | 0.526 | 0.597 |
| LumB | 0.502 | 0.504 | 0.518 | 0.570 |
| Basal | 0.504 | 0.504 | 0.513 | 0.646 |
| HER2 | 0.502 | 0.481 | 0.517 | 0.593 |

### Enrichr (FDR < 0.05)
- **Luminal A (22 genes, 5 terms: 3 GO:BP, 2 KEGG, 0 Hallmark):**
  - `Intermediate Filament Organization (GO:0045109)` — `4/72`, Adj.P `1.12 × 10⁻⁴` (`KRT17, KRT14, KRT5, KRT6B`)
  - `Supramolecular Fiber Organization (GO:0097435)` — `5/339`, Adj.P `1.47 × 10⁻³` (`KRT17, KRT14, KRT5, DSG3, KRT6B`)
  - `Chromosome Organization (GO:0051276)` — `2/47`, Adj.P `4.20 × 10⁻²` (`RNASE7, PXDNL`)
- **Số term đạt FDR < 0.05 ở 3 subtype còn lại:** LumB `65` (17 genes), HER2 `292` (22 genes), Basal `53` (32 genes), ALL `51` (58 genes).

---

## 3. Lệnh chạy lại toàn bộ từ đầu

Chạy tại thư mục gốc repo (`/home/user1/Documents/OSCAR`):

```bash
# 1. Retrain 5 fold BRCA từ report v2 (deterministic) và lưu weights
PYTHONPATH=. .venv/bin/python metrix/token_oscar_retrain_save.py

# 2. Chạy Step 1 (consensus), Step 2 (DE) và vẽ Fig 5.1–5.3
PYTHONPATH=.:bio_material/modules .venv/bin/python metrix/token_oscar_biomarker_figs.py

# 3. Tính gate và vẽ Fig 5.4
PYTHONPATH=.:bio_material/modules .venv/bin/python metrix/token_oscar_gate_fig.py

# 4. Chạy Step 3 (Enrichr, cần mạng)
uv run --with gseapy python metrix/biomarker_step3_enrichment.py \
  --consensus-csv attention_plots/token_oscar_step1_consensus/consensus_features.csv \
  --de-csv attention_plots/token_oscar_step2_de/consensus_with_de.csv \
  --output-dir attention_plots/token_oscar_step3_enrichment \
  --class-names LumA,LumB,HER2,Basal
```
