# HỒ SƠ PHẢN BIỆN & CẨM NANG BẢO VỆ ĐỒ ÁN TỐT NGHIỆP (DEFENSE Q&A BANK)

**Đề tài:** *Multi-omics integration for Cancer Subtyping*  
**Sinh viên:** Hai Duc Tran — Data Science (Elitech), SOICT, HUST  
**Giảng viên hướng dẫn:** Duc-Hau Le  
**Phạm vi:** Mock Defense chuẩn Hội đồng SOICT-HUST + Phản biện tạp chí *Bioinformatics* / *Briefings in Bioinformatics* / *ICLR Workshop*.

---

## MỤC LỤC NHANH — 16 CÂU HỎI "CHÍ MẠNG" THEO 5 NHÓM HỘI ĐỒNG

| Nhóm | Vai trò phản biện | Các đòn tấn công chính |
|---|---|---|
| **Nhóm A** | **Giáo sư ML / Deep Learning** | Q1. Bottleneck hạng 256 vs bùng nổ 8,192 chiều & 16.42M tham số · Q2. Nghịch lý Table 5.2 (Concat 91.44% > mọi baseline, Cross-Attn đơn lẻ lại làm TỤT điểm!) · Q3. Đại số `TokenGatedFusion` thực chất là residual scaling $Z_i + \alpha_i \odot \Delta_i$ & gate kẹt ở $\approx 0.50$ · Q4. Dư thừa $e_i + P$ và vì sao Step 2 dùng $Z_T$ thay vì $F_T$ · Q5. Vì sao giữ $256 \times 32$ khi $128 \times 32$ đạt F1 cao hơn với ít hơn 3.17M tham số? |
| **Nhóm B** | **Chuyên gia Thống kê & Thực nghiệm** | Q6. Thiếu baseline đơn omic (`mRNA-only`) & vòng luẩn quẩn nhãn PAM50/CMS (Label Circularity) · Q7. Thứ tự sinh học THUA Reverse trên BRCA và THUA Mixed trên COADREAD (Table 5.3) · Q8. Công bằng so sánh baseline (`moBRCA-net`, `MOGONET`) & thiếu kiểm định ý nghĩa thống kê (1 seed) · Q9. Nhiễu chọn `chosen_epoch` trên COADREAD ($N_{\text{val}} \approx 35$, ~5 mẫu/lớp) & lệch số bước gradient |
| **Nhóm C** | **Chuyên gia Tin sinh học & Ung thư học** | Q10. Điểm yếu Vanilla Gradient (`|∂logit/∂x|`) & thiên vị quy mô assay (RPPA 223 vs mRNA 19,580) · Q11. Double-dipping (Post-selection inference) & FDR nội bộ thay vì toàn hệ gen · Q12. Tại sao Enrichr Luminal A ra *Staphylococcus aureus infection* và 0 Hallmark? · Q13. Đề cử *TBC1D31* có thật sự mới hay chỉ là co-amplification 8q24 cạnh *MYC*? |
| **Nhóm D** | **Phản biện Tính mới (Novelty 2024–2026)** | Q14. So với `CrossAttOmics` (Bioinformatics 2025) & `DoGMA` (2026) — hai model ĐÃ làm directed cross-attention theo Central Dogma — luận văn có gì mới? · Q15. So với `Pathformer` (Bioinformatics 2024) & `MOT` (Bioinformatics 2023) — tại sao dùng latent tokens vô danh thay vì biological pathway tokens? |
| **Nhóm E** | **Kiểm tra Nhất quán Báo cáo & Triển khai** | Q16. Không có external validation (METABRIC / CPTAC), không xử lý missing omics, và không có confusion matrix |

---

## NHÓM A: HỘI ĐỒNG MACHINE LEARNING / DEEP LEARNING

### Q1. [CHÍ MẠNG — Đại số tuyến tính & Over-parameterization]
> **Hội đồng hỏi:**  
> *"Trong Phương trình (3.2) (`OmicToTokens`), em chiếu mỗi omic qua bottleneck $d_{\text{mid}} = 256$, rồi nhân tuyến tính $W_{2,i} \in \mathbb{R}^{8192 \times 256}$ để reshape thành $T=256$ token, mỗi token rộng $M=32$. Về mặt đại số, toàn bộ $256 \times 32 = 8{,}192$ tọa độ nằm trong một không gian con affine có bậc tự do tối đa là 256 — bằng đúng 1 vector 256 chiều! Vậy $T=256$ token này có thật sự mang thêm thông tin không? Chưa kể, trên COADREAD ($N=222$, inner-train chỉ ~142 bệnh nhân), mô hình có tận **16.42 triệu tham số** (riêng RPPA có 153 protein mà em phình lên 8,192 chiều, gấp 53.5 lần, tốn 2.1 triệu tham số chỉ cho $W_{2,P}$). Tại sao không bị overfit chết luôn?"*

- **Điểm yếu thực sự trong luận văn:**
  - Phương trình (3.2) trong `Chapter/3_Methodology.tex` (dòng 68–71): $Z_i = \text{Reshape}_{T \times M}(W_{2,i} h_i + b_{2,i})$ với $h_i \in \mathbb{R}^{256}$. Ma trận $Z_i$ đúng là bị giới hạn hạng thông tin đầu vào bởi $h_i \in \mathbb{R}^{256}$ (không phải 256 nhóm gen độc lập như `Pathformer` hay `CrossAttOmics`).
  - Tổng số tham số 16.42M trên BRCA: riêng $W_{1,G}, W_{1,T}$ đã chiếm $\approx 9.94\text{M}$ ($19273\times 256 + 19580\times 256$), và 3 ma trận $W_{2,i}$ chiếm thêm $3 \times 256 \times 8192 = 6.29\text{M}$ tham số!
- **Bằng chứng từ literature:**
  - *CrossAttOmics* (Beaude et al., *Bioinformatics* 2025, [DOI: 10.1093/bioinformatics/btaf302](https://doi.org/10.1093/bioinformatics/btaf302)) giải quyết vấn đề token bằng cách phân nhóm trực tiếp feature đầu vào (feature-group decomposition).
  - *Pathformer* (Liu et al., *Bioinformatics* 2024, [PMC11139513](https://pmc.ncbi.nlm.nih.gov/articles/PMC11139513/)) dùng mạng thưa ánh xạ gen $\to$ 2,289 biological pathways làm token.
  - Tuy nhiên, trong lý thuyết Transformer (như Perceiver / factorized projection), một lớp linear expansion từ bottleneck $h_i \in \mathbb{R}^{256}$ sang $T$ slot tương đương với việc học **$T$ phép chiếu con khác nhau** $z_{i,t} = W_{2,i}^{(t)} h_i + b_{2,i}^{(t)} \in \mathbb{R}^{32}$ đóng vai trò các *latent views / subspaces* cho attention tương tác phi tuyến ở tầng sau.
- **Cách đỡ đòn (Script 30 giây + Deep-dive):**
  1. **Thừa nhận ngay về mặt đại số (đừng cãi cùn!):** *"Thưa thầy, nhận xét đại số của thầy hoàn toàn chính xác: ngay sau `OmicToTokens`, bậc tự do của $Z_i$ bị chặn bởi $d_{\text{mid}}=256$."*
  2. **Giải thích lý do thiết kế (2 ý sắc bén):**
     - **Thứ nhất — Tránh suy biến Softmax và mở khóa tương tác phi tuyến bậc cao:** Khi $T=1$, cross-attention có đúng 1 key nên $\text{softmax}(QK^\top/\sqrt{d_k}) \equiv 1$ (suy biến thành hằng số không phụ thuộc đầu vào). Khi tách thành $T$ latent slots ($z_{i,t} = W_{2,i}^{(t)} h_i + b_{2,i}^{(t)}$), dù tuyến tính ở đầu ra encoder, nhưng ngay khi đi vào Multi-Head Cross-Attention và 2 tầng Pre-LN Transformer (có softmax, GELU, và token-wise gate), các phép tương tác bậc cao giữa các *subspace* khác nhau của CNV, mRNA và RPPA mới có thể hình thành. **Bằng chứng thực nghiệm nằm ngay ở Table 5.5:** trên $1\times 128$, bỏ self-attention lại *tăng* F1 (+0.94%), còn trên $256\times 32$, bỏ self-attention làm *giảm* F1 (-1.38%).
     - **Thứ hai — Tại sao 16.42M tham số không sập vì overfit:** Hơn 60% số tham số (~10M) nằm ở lớp chiếu đầu vào bắt buộc $W_{1,G}, W_{1,T}$ do input thô có ~40,000 gen (mọi mạng MLP/AE không feature-selection đều có con số này). Bottleneck $d_{\text{mid}}=256$, Dropout 0.4 xuyên suốt mọi block, LayerNorm, Weight Decay và inner-val early stopping (`chosen_epoch` rơi vào 24–57 trên BRCA) đóng vai trò regularization mạnh.
  3. **Điểm cộng trung thực:** *"Đối với RPPA chỉ có 153–223 chiều, đúng là việc dùng chung $T=256$ cho cả RPPA hơi dư thừa tham số; thực tế Table 5.4 cho thấy $128\times 32$ (13.25M) đạt Avg F1 92.41%, thậm chí nhỉnh hơn $256\times 32$. Trong hướng phát triển, việc dùng số token bất đối xứng theo từng omic ($T_P < T_G, T_T$) hoặc sinh token từ gene-set/pathway là cải tiến rất đáng giá."*

---

### Q2. [CHÍ MẠNG — Nghịch lý Table 5.2: Concatenation đã thắng tất cả baseline, còn Cross-Attention đơn lẻ lại làm TỤT điểm!]
> **Hội đồng hỏi:**  
> *"Em nhìn vào Table 5.2 đi:*
> - *`Concatenation` (bỏ sạch cross-attention, self-attention, gates) đã đạt **91.44% Avg F1**, vượt TẤT CẢ 5 baseline bên ngoài (từ MoXGATE 89.64% đến DeepMoIC 85.02%)!*
> - *Thêm toàn bộ khối Ordered Cross-Attention + 2 Gates + 2-layer Transformer 769 token chỉ tăng thêm đúng **+0.74% Avg F1** (nhỏ hơn gấp 3 lần độ lệch chuẩn $\pm 2.15\%$–$2.55\%$)!*
> - *Thậm chí tệ hơn: so dòng 1 (`Concatenation`: **91.44% F1**) với dòng 2 (`Without self-attention`, tức là CÓ Cross-Attention + Cross Gate: **90.80% F1**), việc thêm Ordered Cross-Attention vào mô hình **LÀM TỤT MACRO-F1 TRÊN CẢ 3 TẬP DỮ LIỆU** (BRCA $88.85 \to 88.71$, LUNG $95.87 \to 95.72$, COADREAD $89.61 \to 87.96$)!*
> *Vậy thực chất mô hình của em mạnh nhờ mỗi cái MLP bottleneck + Dropout 0.4 + matched training recipe, chứ Cross-Attention có giúp ích gì đâu?"*

- **Điểm yếu thực sự trong luận văn:**
  - Con số trong `Chapter/5_Numerical_results.tex` Table 5.2 (dòng 69–70) là **100% có thật** và cực kỳ nhạy cảm:
    - `Concatenation`: BRCA `88.85`, LUNG `95.87`, COADREAD `89.61`, Avg `91.44`
    - `Without self-attention (cross + gate)`: BRCA `88.71`, LUNG `95.72`, COADREAD `87.96`, Avg `90.80` (-0.64 F1 so với Concat!)
    - `Without cross-attention (self + gate + pos)`: BRCA `88.48`, LUNG `96.46`, COADREAD `89.21`, Avg `91.38` (-0.06 F1 so với Concat!)
    - `Full`: BRCA `89.93`, LUNG `97.34`, COADREAD `89.26`, Avg `92.18` (+0.74 F1 so với Concat!)
- **Cách đỡ đòn (Đây là câu hỏi quyết định điểm A hay B — phải trả lời cực kỳ thông minh và trung thực!):**
  1. **Thừa nhận thẳng thắn (đúng tinh thần trung thực khoa học của luận văn):** *"Thưa thầy, đây chính là phát hiện quan trọng nhất trong Chương 5 và Chương 6 của em. Em cố tình báo cáo đầy đủ dòng `Concatenation` để không thổi phồng vai trò của attention: đúng là bộ mã hóa token hóa (`OmicToTokens`) + cấu hình huấn luyện chuẩn hóa đã gánh phần lớn cách biệt so với 5 baseline (91.44% vs 89.64%)."*
  2. **Giải thích cơ chế tại sao `Cross-only` (90.80%) và `Self-only` (91.38%) đều kém hơn `Full` (92.18%) — Hiệu ứng hiệp đồng (Synergy) và cách readout:**
     - Ở biến thể `Without self-attention (cross + gate)`, do bỏ Transformer và `[CLS]` token, đầu ra $T=256$ token của mỗi omic bị **mean-pool phẳng** trước khi đưa vào classifier. Mean-pooling làm mất cấu trúc phân hóa giữa các token mà cross-attention vừa tạo ra (mọi token bị cộng trung bình đều nhau, gây nhiễu đặc biệt trên COADREAD $N=222$).
     - Ở biến thể `Without cross-attention (self + gate + pos)`, Transformer self-attention giúp tăng LUNG (+0.59 F1) nhưng lại *giảm* BRCA (88.48% so với 88.85%) vì thiếu bước căn chỉnh chéo CNV $\to$ mRNA và mRNA $\to$ RPPA trước khi trộn 769 token.
     - **Chỉ khi kết hợp cả hai (`Full`)**, cross-attention chuẩn bị token đã làm giàu theo chiều omic rồi Transformer + `[CLS]` mới tổng hợp chọn lọc thay vì mean-pool mù quáng: kết quả là `Full` vượt cả `Concat` (+1.08 F1 trên BRCA, +1.47 F1 trên LUNG, +1.25% Avg Acc) đồng thời **giảm mạnh phương sai trên COADREAD** ($\pm 4.79\% \to \pm 2.55\%$!).
  3. **Chốt câu ghi điểm:** *"Điều này cho thấy hai module không hoạt động cộng tính đơn giản mà cần đi cặp với nhau — cross-attention cần cơ chế tổng hợp có trọng số (`[CLS]` + self-attention) để khai thác hết thông tin token-level, và ngược lại."*

---

### Q3. [CHÍ MẠNG — Bắt lỗi Đại số Phương trình (3.5) và Gate kẹt quanh 0.50]
> **Hội đồng hỏi:**  
> *"Em xem lại Phương trình (3.3)–(3.5) trong Chương 3 và giải thích ở Mục 4.2:*
> - *Trong `TokenCrossAttention`, đầu ra $H_i$ ĐÃ CÓ kết nối residual từ $Z_i$: $H_i = Z_i + \Delta_i$ (trong đó $\Delta_i = \text{MHA}(\cdot) + \text{FFN}(\cdot)$).*
> - *Sau đó em đưa vào `TokenGatedFusion`: $F_i = \alpha_i \odot H_i + (1 - \alpha_i) \odot Z_i$. Thay $H_i = Z_i + \Delta_i$ vào, ta có chính xác:*
> $$F_i = Z_i + \alpha_i \odot \Delta_i$$
> *Vậy câu viết ở Chương 4 rằng 'values near 1 ignore the encoder token $Z_i$' là **SAI về mặt đại số** — ngay cả khi $\alpha_i = 1$, $Z_i$ vẫn được giữ nguyên 100%!*
> *Hơn nữa, ở Hình 5.4 (Section 5.6), giá trị trung bình của gate trên mRNA là **0.502–0.504** và Proteomics là **0.513–0.526**. $\sigma(0) = 0.500$! Em có chắc gate thực sự học được trọng số phụ thuộc bệnh nhân không, hay MLP gate với dropout 0.4 chỉ đang đầu ra loanh quanh 0 (nên $\alpha \approx 0.5$), tương đương với một hằng số cố định `0.5`?"*

- **Điểm yếu thực sự trong luận văn:**
  - **Lỗi đại số trong `Chapter/4_Theoretical_Analysis.tex` (dòng 18):** Đúng là $H_i = Z_i + \Delta_i \implies F_i = \alpha_i(Z_i + \Delta_i) + (1-\alpha_i)Z_i = Z_i + \alpha_i \odot \Delta_i$. Vậy khi $\alpha_i \to 0$, $F_i \to Z_i$ (bỏ cập nhật cross-attention); khi $\alpha_i \to 1$, $F_i \to Z_i + \Delta_i = H_i$ (nhận 100% cập nhật cross-attention *cộng với* $Z_i$, KHÔNG hề vứt bỏ $Z_i$!).
  - **Về $\alpha \approx 0.50$:** Luận văn không có thí nghiệm so sánh `learned gate` với `fixed α = 0.5` (hoặc `LayerScale`), cũng chưa báo cáo phân phối/độ lệch chuẩn của $\alpha_{t,m}$ trên từng tọa độ bên trong 1 bệnh nhân.
- **Cách đỡ đòn:**
  1. **Nhận lỗi đại số ngay lập tức (thể hiện bản lĩnh toán học cực tốt):** *"Thưa thầy, thầy bắt rất đúng ạ! Do `TokenCrossAttention` dùng Pre-LN residual ($H_i = Z_i + \Delta_i$), biểu thức `TokenGatedFusion` rút gọn đại số đúng thành $F_i = Z_i + \alpha_i \odot \Delta_i$. Vậy thực chất đây là một **input-dependent residual scaling gate** (giống LayerScale/Highway residual) điều tiết độ lớn của phần cập nhật $\Delta_i$ thay vì bỏ hoàn toàn $Z_i$ khi $\alpha_i \to 1$. Em xin tiếp thu và chỉnh lại diễn đạt ở Mục 4.2 cho chính xác tuyệt đối."*
  2. **Trả lời câu hỏi "Gate có đang chết ở $\sigma(0) = 0.5$ không?":**
     - **Bằng chứng 1 (từ Hình 5.4 Panel B):** Với Proteomics, tuy *mean* là 0.513–0.526 nhưng **57.0%–64.6%** số tọa độ có $\alpha > 0.5$ (cao nhất ở Basal-like 64.6% so với Luminal B 57.0%, với độ lệch chuẩn giữa các fold nhỏ 0.01–0.05). Nếu gate bị kẹt ngẫu nhiên quanh 0, tỷ lệ $> 0.5$ phải dao động đối xứng quanh 50% ở mọi phân nhóm chứ không dịch chuyển nhất quán lên 57%–65% và khác biệt theo subtype như vậy.
     - **Bằng chứng 2 (từ Ablation Table 5.2):** Nếu gate vô dụng (chỉ như $\alpha=1$ khi bỏ gate), thì bỏ `cross gate` không nên ảnh hưởng nhiều. Thực tế, bỏ `cross gate` làm giảm BRCA từ **89.93% xuống 88.22% (-1.71 F1)** và LUNG từ **97.34% xuống 96.17% (-1.17 F1)**; bỏ `self gate` làm tăng vọt độ lệch chuẩn trên BRCA ($\pm 2.15 \to \pm 3.63$) và COADREAD ($\pm 2.55 \to \pm 5.31$).
     - **Bổ sung trung thực:** *"Tuy nhiên, em đồng ý rằng để tách bạch hoàn toàn giữa hiệu ứng 'co giãn hệ số residual 0.5 cố định' và 'trọng số động phụ thuộc bệnh nhân', phép đối chứng với hằng số cố định $\alpha = 0.5$ và thống kê phương sai nội bộ giữa các tọa độ là thử nghiệm bổ sung rất xác đáng."*

---

### Q4. [Thiết kế kiến trúc — Wiring Step 2 dùng $Z_T$ thay vì $F_T$ & Dư thừa $e_i + P$]
> **Hội đồng hỏi:**  
> *"Có hai chi tiết rất lạ trong kiến trúc Chương 3:*
> 1. *Em tuyên bố mô hình theo chuỗi Central Dogma $\text{CNV} \to \text{mRNA} \to \text{Proteomics}$. Nhưng ở Step 2 (Mục 3.5 & Hình 3.3), Proteomics lại query vào **$Z_T$ gốc** chứ KHÔNG PHẢI $F_T$ (mRNA đã được CNV cập nhật)! Như vậy đâu phải chuỗi truyền tiếp (transitive cascade $G \to T \to P$), mà là 2 cạnh rời rạc song song ($G \to T$ và $T \to P$)? Em có ablation so sánh $Z_T$ với $F_T$ chưa?*
> 2. *Ở Phương trình (3.6): $S_0 = [u_0;\; F_T + e_T;\; Z_G + e_G;\; F_P + e_P] + P$. Vì mỗi omic luôn nằm ở một dải vị trí cố định trong $769$ token, ma trận $P \in \mathbb{R}^{769 \times 32}$ đã có riêng 1 vector cho từng vị trí rồi — cộng thêm $e_i$ chẳng phải dư thừa tuyến tính (rank-redundant) sao? Hơn nữa, $256$ token sinh ra từ reshape MLP đâu có thứ tự không gian như câu văn, cần gì positional embedding?"*

- **Cách đỡ đòn:**
  1. **Về việc Step 2 dùng $Z_T$ thay vì $F_T$:**
     - **Lý do kỹ thuật & sinh học:** Trong sinh học phân tử, dịch mã protein trực tiếp đọc bản sao mRNA hiện diện ($Z_T$); còn CNV ảnh hưởng đến protein chủ yếu gián tiếp qua mức biểu hiện mRNA đo được (mà $Z_T$ thực đo đã chứa sẵn hậu quả của CNV rồi!). Nếu dùng $F_T$ (vốn đã pha thêm nhiễu từ 24,776 biến CNV độ nhiễu cao) để cập nhật cho 153–223 biến RPPA sạch hơn, lỗi dự đoán từ CNV có thể khuếch đại chuỗi (error propagation) sang Proteomics.
     - **Thừa nhận thẳng:** *"Đúng là về mặt đồ thị tính toán trong khối cross-attention, dùng $Z_T$ tạo thành 2 bước có hướng chia sẻ nguồn trung gian ($G \to T$ và $T \to P$) trước khi 2 tầng Transformer phía sau mới ghép toàn bộ $(F_T, Z_G, F_P)$ lại. Em chưa đưa bảng ablation trực tiếp giữa $Z_T$ vs $F_T$ vào báo cáo cuối cùng — đây là điểm cần bổ sung kiểm chứng định lượng."*
  2. **Về $e_i + P$ và positional embedding cho token reshape:**
     - **Tại sao cần positional embedding $P$ dù là token reshape:** Đúng là $256$ token không có thứ tự tuyến tính như từ ngữ, nhưng mỗi chỉ số $t \in \{1,\dots,256\}$ đại diện cho một **slot đặc trưng khác nhau** sinh bởi khối dòng thứ $t$ của $W_{2,i}$. Vì Self-Attention có tính bất biến hoán vị (permutation-equivariant), nếu không có embedding riêng theo vị trí slot, Transformer không phân biệt được token slot #5 của mRNA với token slot #120 của mRNA khi độ lớn vector giống nhau (tương tự như vai trò của learned positional embeddings trong ViT/Perceiver).
     - **Thừa nhận sự dư thừa đại số của $e_i + P$:** *"Về mặt biểu diễn tối ưu, đúng là $e_i + P[\text{pos}]$ có thể gộp thành một bảng tham số duy nhất; giữ tách biệt $e_i$ (shared bias cho cả 256 token cùng omic) và $P$ (fine-grained per-slot offset) là một dạng tham số hóa dư thừa nhẹ (over-parameterized inductive bias) tương tự BERT (`token_type_embeddings` + `position_embeddings`), giúp gradient ở cấp omic cập nhật nhanh hơn trong giai đoạn đầu huấn luyện."*

---

### Q5. [Tại sao chọn mặc định $256 \times 32$ khi $128 \times 32$ vừa cao F1 hơn vừa nhẹ hơn?]
> **Hội đồng hỏi:**  
> *"Nhìn Table 5.4 (`tab:token-sensitivity`): cấu hình $128 \times 32$ đạt Avg Macro-F1 **92.41%** (cao hơn cả $256 \times 32$ là **92.18%**), cao hơn cả trên BRCA (89.94% vs 89.93%) và vượt hẳn trên COADREAD (**90.53% vs 89.26%**, +1.27 F1!), lại ít hơn tận **3.17 triệu tham số** (13.25M vs 16.42M) và giảm chi phí self-attention $O(L^2)$ đi **4 lần** ($385^2$ vs $769^2$)! Vậy tại sao em vẫn chọn $256 \times 32$ làm default?"*

- **Cách đỡ đòn:**
  - *"Thưa thầy, cấu hình $256 \times 32$ được cố định **trước** khi chạy toàn bộ bảng so sánh chính (Table 5.1) và bảng độ nhạy (Table 5.4) để tránh lỗi chọn siêu tham số dựa trên kết quả test (post-hoc cherry-picking). Nếu sau khi thấy Table 5.4 mà em đổi kết quả báo cáo chính sang $128 \times 32$, em sẽ vô tình làm lệch đánh giá trên tập test.*
  - *Mục đích của Table 5.4 là chứng minh mô hình **ổn định trên một vùng rộng** ($T \ge 128$ đều hội tụ trong biên 0.12% Acc và 0.23% F1), không bị phụ thuộc mong manh vào một điểm cấu hình duy nhất. Thực tế triển khai nếu tối ưu tài nguyên tính toán và bộ nhớ, $128 \times 32$ đúng là điểm cân bằng (Pareto-optimal) tốt hơn."*

---

## NHÓM B: HỘI ĐỒNG THỐNG KÊ & PHƯƠNG PHÁP THỰC NGHIỆM

### Q6. [CHÍ MẠNG NHẤT — Thiếu Single-Omic Baseline (`mRNA-only`) & Vòng luẩn quẩn nhãn PAM50/CMS]
> **Hội đồng hỏi:**  
> *"Đề tài của em là **Multi-omics integration**, nhưng em KHÔNG HỀ có bất kỳ baseline đơn omic nào (`mRNA-only`, `CNV-only`, `RPPA-only`) cũng như không có baseline Machine Learning cổ điển (L2 Logistic Regression / ElasticNet / SVM)!*
> *Em có biết rằng:*
> 1. *Nhãn **PAM50** trên BRCA (Parker et al., 2009) được định nghĩa **100% từ biểu hiện mRNA** của 50 gen? Và nhãn **CMS1–CMS4** trên COADREAD (Guinney et al., *Nature Medicine* 2015) cũng được định nghĩa từ **gene expression (mRNA)**?*
> 2. *Trong paper **Cascianelli et al. (Scientific Reports 2020)**, chỉ một mô hình **Logistic Regression có điều chuẩn trên RNA-seq đơn lẻ** đã đạt ~88–90% concordance với PAM50! Trong **M²CGCN (Briefings in Bioinformatics 2025)** — bài mà em có trích dẫn trong tài liệu tham khảo — tác giả thậm chí phải **LOẠI BỎ 48 gen PAM50 khỏi dữ liệu mRNA** để tránh mô hình học vẹt trực tiếp định nghĩa nhãn! Còn em giữ nguyên cả 19,580 gen mRNA (nhìn Hình 5.1: `ERBB2`, `ESR1`, `GRB7`, `FOXC1`, `KRT5`, `KRT14`, `KRT17`, `NDC80` đều là gen lõi PAM50!).*
> 3. *Systematic review gần đây trên *Computational and Structural Biotechnology Journal* ([DOI: 10.34133/csbj.0243](https://spj.science.org/doi/10.34133/csbj.0243)) chỉ ra chỉ 61.4% nghiên cứu chứng minh được việc thêm omic khác cải thiện so với mRNA đơn lẻ, và *OmicsFootPrint* (*NAR* 2024, [PMC11602161](https://pmc.ncbi.nlm.nih.gov/articles/PMC11602161/)) thấy khác biệt giữa multi-omics và expression-only trên BRCA **không có ý nghĩa thống kê**!*
> *Vậy em lấy gì chứng minh rằng mô hình tích hợp 3 omics của em thực sự tốt hơn chạy chính `OmicToTokens` trên **mỗi mRNA**, chứ không phải đang tốn 16 triệu tham số chỉ để học lại công thức PAM50/CMS từ mRNA?"*

- **Tại sao câu này cực kỳ nguy hiểm:** Đây là câu hỏi kinh điển của mọi reviewer tạp chí tin sinh học hàng đầu (Nature Methods / Bioinformatics / Briefings in Bioinformatics). Không có `mRNA-only` ablation và không loại PAM50 genes là 2 lỗ hổng thực sự.
- **Cách đỡ đòn (Bình tĩnh, sắc sảo, không bao biện):**
  1. **Thừa nhận thẳng giới hạn thực nghiệm:** *"Thưa thầy, đây là một câu hỏi rất sắc và em hoàn toàn đồng ý: việc thiếu bảng đối chứng đơn omic (`mRNA-only`, `CNV-only`, `RPPA-only`) và một baseline tuyến tính chuẩn hóa (như Ridge/L2 Logistic Regression) là hạn chế rõ ràng trong bộ thí nghiệm hiện tại, em cần bổ sung ngay trong nghiên cứu tiếp theo."*
  2. **Làm rõ phạm vi & lý do giữ nguyên toàn bộ bộ gen (để khớp công bằng với các paper benchmark):**
     - **Về mặt giao thức so sánh:** Toàn bộ 5 baseline trong Table 5.1 (MOGONET, MoGCN, DeepMoIC, moBRCA-net, MoXGATE) và bản thân đề xuất đều nhận cùng một không gian đặc trưng (không lọc bỏ PAM50) để đánh giá công bằng bài toán **chọn kiến trúc fusion** trong thiết lập đa omic chuẩn của tài liệu (tương tự MoGCN 2022, DeepMoIC 2024, MoXGATE 2025).
     - **Bằng chứng gián tiếp trong chính dữ liệu luận văn rằng mô hình KHÔNG chỉ dùng mRNA:**
       - Nhìn Hình 5.1 & 5.2 (BRCA consensus saliency): trên cả 4 subtype, **Proteomics (RPPA)** và **CNV** xuất hiện dày đặc trong top-12 consensus ổn định 5 fold (ví dụ HER2_pY1248 $d=+2.61$, HER2 $d=+2.48$, ERALPHA $d=+1.25$ / $-2.27$, GATA3 $d=+0.87$, CNV 17q12 *GRB7/TCAP/STARD3/IKZF3/CDK12* $d=+1.58\text{–}1.94$).
       - Nhìn Hình 5.4 (Gate trên BRCA): nhánh Proteomics mở gate $\alpha > 0.5$ trên 57%–65% tọa độ, và Table 5.2 cho thấy khi bỏ Cross-Attention (CNV$\to$mRNA, mRNA$\to$RPPA), BRCA F1 giảm **-1.45 điểm** ($89.93\% \to 88.48\%$), cho thấy tín hiệu từ CNV và RPPA thực sự đóng góp bổ trợ lên quyết định cuối cùng, đặc biệt ở ranh giới khó như Luminal A vs Luminal B và HER2.
     - **Về ý nghĩa thực tế của việc lọc bỏ PAM50 (như M²CGCN 2025):** *"Nhận xét về vòng luẩn quẩn nhãn PAM50/CMS rất sâu sắc. Thử nghiệm loại trừ 50 gen PAM50 khỏi mRNA để ép mô hình khai thác tín hiệu bù trừ từ CNV và RPPA — cũng như kiểm chứng trên các bài toán lâm sàng không định nghĩa bằng mRNA như sống còn (OS/PFS) hoặc đáp ứng thuốc — là bước kiểm tra giá trị thực sự của đa omic."*

---

### Q7. [CHÍ MẠNG — Thứ tự sinh học THUA Reverse trên BRCA và THUA Mixed trên COADREAD!]
> **Hội đồng hỏi:**  
> *"Cả Chương 1, 2, 3, 4 em đặt trọng tâm vào giả thuyết **Biological Central-Dogma Ordering (CNV $\to$ mRNA $\to$ Proteomics)**. Thế nhưng mở Table 5.3 (`tab:order`) ra:*
> - *Trên **BRCA** (tập duy nhất em phân tích sinh học sâu!): thứ tự **Ngược (Reverse: Proteomics $\to$ mRNA $\to$ CNV)** đạt **90.85% F1 / 91.98% Acc**, **THẮNG** thứ tự sinh học (89.93% F1 / 91.39% Acc) tận **+0.92 F1**!*
> - *Trên **COADREAD**: thứ tự **Mixed (mRNA $\to$ CNV $\to$ Proteomics)** đạt **90.05% F1 / 91.46% Acc**, **THẮNG** thứ tự sinh học (89.26% F1 / 91.00% Acc) tận **+0.79 F1**!*
> - *Trung bình 3 tập, Biological (92.18%) chỉ hơn Reverse (92.11%) đúng **0.07%** — nhỏ hơn độ lệch chuẩn gấp 30 lần!*
> *Vậy giả thuyết chính của luận văn **BỊ PHỦ ĐỊNH (FALSIFIED)** rồi đúng không? Nếu thứ tự ngược còn tốt hơn trên BRCA, tại sao mô hình vẫn giữ tên gọi/định hướng 'ordered central-dogma'?"*

- **Điểm mạnh bất ngờ của luận văn:** Chính tác giả đã viết rất trung thực ở `Chapter/5_Numerical_results.tex` dòng 117 và `Chapter/6_Conclusions.tex` dòng 9 rằng *"the direction hypothesis is therefore neither supported nor refuted by these runs"*. Đây là lá chắn cực tốt — đừng bao giờ nói ngược lại báo cáo!
- **Cách đỡ đòn (Chuyển thế bị động thành điểm cộng về tư duy khoa học chuẩn mực):**
  1. **Khẳng định bản chất khoa học của luận văn:** *"Thưa thầy, trong luận văn em chủ động thiết lập thứ tự sinh học như một **giả thuyết thực nghiệm có thể kiểm chứng (testable inductive bias)** chứ không mặc định nó đúng. Em chạy đủ cả **6/6 hoán vị** dưới cùng giao thức khóa chặt (Table 5.3) chính để kiểm chứng nghiêm túc giả thuyết này."*
  2. **Trả lời trực diện: Giả thuyết không được xác nhận mạnh, nhưng cũng chưa bị phủ định — và kết quả âm tính/trung tính này có giá trị khoa học rõ ràng:**
     - **Tại sao Reverse lại mạnh trên BRCA?** Trong phân loại ngược ($P \to T \to G$), CNV (20,000 biến thô, rất nhiễu và rời rạc) lại đóng vai trò Query đi lọc thông tin từ mRNA sạch hơn, và mRNA lại được định hướng bởi 223 protein chức năng RPPA vốn rất gần kiểu hình (như ERALPHA, HER2_pY1248, GATA3). Về mặt suy diễn Bayes (diagnostic inference $P(\text{subtype} \mid X)$), chiều suy luận từ *kiểu hình/protein ngược về kiểu gen* hoàn toàn có thể hữu ích cho phân lớp, khác với chiều nhân quả sinh học $DNA \to RNA \to Protein$!
     - **Kết luận khoa học rút ra:** Đóng góp chính được thực nghiệm ủng hộ vững chắc là **giao diện token hóa (`OmicToTokens`) + kiến trúc Attention/Gating đa omic**, còn việc áp đặt cứng thứ tự Central Dogma đem lại lợi ích trung bình rất nhỏ (+0.07% đến +0.82% F1, nằm trong dao động fold) và phụ thuộc từng loại ung thư.
  3. **Liên hệ ngay với literature 2026 (`DoGMA`, arXiv:2608.08148):** *"Thực tế, paper `DoGMA` (2026) cũng gặp đúng câu hỏi này và chọn hướng **soft directional bias** ($B_{\text{dogma}} = A + R_{ij}$: phạt nhẹ cạnh ngược nhưng vẫn cho mô hình học residual) thay vì khóa cứng một chiều tuyệt đối như thiết kế hiện tại của em. Đây chính là hướng mở rộng (ii) em đã đề xuất trong Chương 6."*

---

### Q8. [Công bằng của Baseline Comparison & Thiếu Kiểm định Thống kê (Single Seed)]
> **Hội đồng hỏi:**  
> *"Em so sánh với 5 baseline (Table 5.1), nhưng:*
> 1. *Ở Chương 6 (mục v), em thừa nhận các baseline chưa được tune hyperparameter theo cùng budget. Đặc biệt: `moBRCA-net` (Choi & Chae, 2023) vốn thiết kế cho **mRNA + DNA methylation + miRNA** (không phải CNV + RPPA); `MOGONET` (Wang et al., 2021) vốn yêu cầu bước **chọn lọc đặc trưng ANOVA/variance** trước khi dựng đồ thị tương đồng bệnh nhân — em đổ nguyên 24,776 đặc trưng CNV thô vào đồ thị cosine của MOGONET khiến nó tụt thê thảm trên BRCA (74.55% F1), trong khi trên COADREAD MOGONET vẫn **thắng** mô hình của em (91.37% vs 89.26%)!*
> 2. *Toàn bộ thí nghiệm chỉ chạy **1 seed duy nhất (seed = 42)** và **KHÔNG CÓ BẤT KỲ KIỂM ĐỊNH THỐNG KÊ NÀO** (p-value) cho 5 fold. Theo **Nadeau & Bengio (Machine Learning 2003)** và **Demšar (JMLR 2006)**, 5 fold trong CV có tập huấn luyện chồng lấn 75% nên không độc lập — ngay cả khi chênh lệch +3.11% F1 trên BRCA ($\pm 2.15$ vs $\pm 2.69$), làm sao em biết đó không phải nhiễu ngẫu nhiên?"*

- **Bằng chứng literature:**
  - Nadeau & Bengio (2003), *Machine Learning* 52:239–281 ([PDF](https://jjcurtin.github.io/book_iaml/pdfs/nadeauC2003.pdf)): Phương sai hiệu hai mô hình trên $k$-fold CV phải hiệu chỉnh bằng hệ số $(1/k + n_{\text{test}}/n_{\text{train}}) = (1/5 + 1/4) = 0.45$ (thay vì $1/5 = 0.20$), tức độ lệch chuẩn hiệu bị khuếch đại lên $\sqrt{0.45/0.20} = 1.5\times$, với $df = 4$!
  - Demšar (2006), *JMLR* 7:1–30 ([PDF](https://jmlr.org/papers/volume7/demsar06a/demsar06a.pdf)): Đề xuất Wilcoxon signed-rank / Friedman test trên nhiều tập dữ liệu (nhưng với chỉ 3 tập dữ liệu thì lực kiểm định phi tham số không đủ để đạt $p < 0.05$).
- **Cách đỡ đòn:**
  1. **Về tính công bằng với baseline:**
     - Nhấn mạnh ưu điểm của giao thức matched 5-fold: tất cả 6 phương pháp đều dùng **cùng chia fold, cùng đặc trưng đầu vào, cùng inner-val 20% chọn epoch, không rò rỉ tập test** — tránh hiện tượng phổ biến trong nhiều bài báo là chép số cũ từ paper khác (vốn dùng tập con bệnh nhân hoặc chia fold khác nhau).
     - **Về MOGONET & MoGCN:** Thực tế, trong bảng này các baseline đồ thị (MOGONET, MoGCN, DeepMoIC) thậm chí còn có lợi thế **transductive** (xây dựng đồ thị trên cả đặc trưng train + test, chỉ che nhãn test), trong khi mô hình đề xuất là hoàn toàn **inductive** (dự đoán độc lập từng bệnh nhân mới). Dù vậy, em công nhận việc giữ nguyên toàn bộ 40,000 đặc trưng không qua ANOVA pre-filtering bất lợi hơn cho các phương pháp dựa trên khoảng cách cosine (như MOGONET trên BRCA), và em đã ghi rõ cần đánh giá thêm dưới ngân sách tuning đồng nhất ở Chương 6.
     - **Về việc MOGONET thắng trên COADREAD (91.37% vs 89.26%):** Với $N=222$ (mỗi fold test chỉ ~44 bệnh nhân, 1 bệnh nhân đổi nhãn = 2.3% Acc), mô hình đồ thị tận dụng truyền tin lân cận giữa bệnh nhân (transductive neighborhood smoothing) có lợi thế rõ rệt ở chế độ cực ít mẫu, còn mô hình đề xuất vẫn đứng thứ 2 và vượt MoXGATE (+3.17 F1), MoGCN (+4.87 F1) với độ ổn định cao ($\pm 2.55\%$).
  2. **Về single-seed & p-value:**
     - *"Em hoàn toàn đồng ý với lưu ý của thầy về Nadeau–Bengio (2003). Chính vì nhận thức rõ phương sai chồng lấn trong 5-fold CV và việc chỉ chạy 1 seed, toàn bộ Chương 4 (Mục 4.4) và Chương 5 của em đều **không tuyên bố ý nghĩa thống kê** cho các khoảng chênh tương đương độ lệch chuẩn (như +3.11 trên BRCA hay +0.07–0.82 ở bảng thứ tự), mà diễn giải trung thực ở mức mô tả (descriptive), đồng thời đề xuất chạy đa seed + kiểm định cặp hiệu chỉnh (corrected resampled t-test) ở hướng phát triển."*

---

### Q9. [Rủi ro kỹ thuật trong Nested 5-fold CV — Inner-Val COADREAD chỉ 35 bệnh nhân & Lệch số bước cập nhật]
> **Hội đồng hỏi:**  
> *"Xem lại Mục 3.7 (Matched Nested 5-Fold Protocol):*
> 1. *Trên COADREAD ($N=222$), outer-train có 178 bệnh nhân. Em tách 20% làm inner-val tức là chỉ có **35 bệnh nhân** chia cho **4 lớp CMS** — lớp CMS1 và CMS3 mỗi lớp chỉ có đúng **5 bệnh nhân**! Chọn `first epoch attaining max inner-val macro-F1` trên 5 bệnh nhân/lớp cực kỳ nhiễu (1 bệnh nhân nhảy nhãn làm Macro-F1 vọt ~5%).*
> 2. *Sau đó em đem đúng `chosen_epoch` ($e_k^*$) retrain trên 100% outer-train với cùng `batch_size = 32` và không có LR scheduler. Khi dữ liệu tăng 25% ($80\% \to 100\%$), mỗi epoch sẽ có **nhiều hơn 25% số bước cập nhật gradient** — vậy $e_k^*$ tối ưu ở inner split chưa chắc tương ứng cùng điểm hội tụ trên outer split!"*

- **Cách đỡ đòn:**
  - *"Thưa thầy, nhận xét này rất tinh tế về mặt tối ưu hóa thực nghiệm:*
    - *Về vấn đề inner-val nhỏ trên COADREAD (35 mẫu): đúng là chọn `first-max` trên 1 split 20% có phương sai cao — điều này giải thích trực tiếp tại sao độ lệch chuẩn giữa các fold trên COADREAD lớn hơn hẳn BRCA/LUNG (ở DeepMoIC lên tới $\pm 10.01\%$). Để giảm nhiễu mà vẫn tuyệt đối chống rò rỉ tập test, cải tiến chuẩn là dùng **inner 3-fold hoặc 5-fold CV** thay vì 1 holdout 20%, hoặc dùng **EMA / validation loss smoothing** thay vì hard max macro-F1.*
    - *Về chênh lệch ~25% số gradient steps khi retrain: giữ nguyên số epoch (thay vì giữ số optimization steps) là quy ước phổ biến, nhưng đúng là chuẩn xác hơn nên quy đổi theo **tổng số gradient steps** ($S_k^* = e_k^* \times \lceil N_{\text{inner}} / B \rceil$) hoặc dùng ensemble 5 inner models thay vì retrain đơn lẻ. Quan trọng nhất là quy tắc này được áp dụng **đồng nhất tuyệt đối cho cả mô hình đề xuất lẫn toàn bộ 5 baseline và mọi biến thể ablation**, nên không tạo thiên vị có lợi cho riêng mô hình của em."*

---

## NHÓM C: HỘI ĐỒNG TIN SINH HỌC & UNG THƯ HỌC

### Q10. [CHÍ MẠNG — Giới hạn của Vanilla Gradient Saliency & Thiên vị chiều dữ liệu RPPA vs mRNA]
> **Hội đồng hỏi:**  
> *"Em dùng **Vanilla Input Gradient** ($\text{Mean}\,|\partial \text{logit}/\partial x|$) để xếp hạng gen quan trọng trên BRCA. Nhưng trong lĩnh vực XAI:*
> 1. *Theo **Sundararajan et al. (ICML 2017 — Integrated Gradients)** và **Adebayo et al. (NeurIPS 2018 — Sanity Checks for Saliency Maps)**, vanilla gradient chịu lỗi **bão hòa gradient (gradient saturation)**, nhạy nhiễu cục bộ ở mạng phi tuyến sâu (GELU/LayerNorm/Transformer), và em chưa chạy bất kỳ sanity check nào (như model parameter randomization hay label permutation) hay so sánh với **Integrated Gradients / GradientSHAP / DeepLIFT**.*
> 2. *Em chuẩn hóa từng omic bằng `StandardScaler`, nhưng **RPPA chỉ có 223 đặc trưng** trong khi **mRNA có 19,580 và CNV có 19,273** (gấp ~88 lần!). Về mặt toán học, khi chiếu vào cùng không gian bottleneck $d_{\text{mid}}=256$, mỗi đặc trưng RPPA trung bình nhận gradient lớn hơn nhiều lần so với một gen mRNA (vì năng lượng gradient không bị chia đều cho 20,000 chiều tương quan nhau). Đây có phải lý do thực sự khiến RPPA áp đảo top features trong Hình 5.1 không?"*

- **Bằng chứng literature:**
  - Adebayo et al. (2018), *NeurIPS*, [arXiv:1810.03292](https://arxiv.org/html/1810.03292): Vanilla gradients vượt qua được parameter/label randomization checks, nhưng **không tự động đảm bảo ý nghĩa sinh học** và kém ổn định hơn IG/Input×Gradient trong vùng bão hòa.
  - *COAT* (Cross-Omics Attention Transformer for PAM50, [Semantic Scholar PDF](https://pdfs.semanticscholar.org/0c2f/f441c4738c4fa44f44b9fc6dc0d2c584f22f.pdf)) dùng **GradientSHAP**; *CrossAttOmics* (2025) dùng **LRP (Layer-wise Relevance Propagation)**; *Pathformer* (2024) dùng **SHAP + pathway attention weights**.
- **Cách đỡ đòn:**
  1. **Trả lời ngay về thiên vị kích thước assay (223 vs 19,580) — CHỨNG MINH EM ĐÃ TRÁNH ĐƯỢC LỖI NÀY!**
     - *"Thưa thầy, đây là điểm em đã chủ động phòng tránh trong thiết kế giao thức (Mục 3.8): em chọn **top-20 RIÊNG TRONG TỪNG OMIC** (per-omic top 20 in $\ge 3/5$ folds) chứ **KHÔNG xếp hạng gộp chung** thang đo gradient giữa RPPA, mRNA và CNV! Vì vậy độ lớn tuyệt đối của gradient giữa các assay khác nhau không làm RPPA đè bẹp mRNA/CNV ở bước lọc ứng viên (kết quả 109 consensus gồm đầy đủ cả 3 lớp omic). Tuy nhiên trên Hình 5.1 (top-12 hỗn hợp để minh họa), đúng là độ lớn gradient thô trên trục x có thiên vị tự nhiên về phía RPPA ít chiều hơn — nhận xét của thầy rất xác đáng."*
  2. **Trả lời về Vanilla Gradient vs IG/SHAP & Sanity Checks:**
     - *"Em chọn vanilla $|\partial \text{logit}/\partial x|$ vì tốc độ tính nhanh trên 40,000 chiều × 5 fold và đã khắc phục một phần nhiễu cục bộ bằng **điều kiện đồng thuận 3/5 fold** + kiểm tra chéo với **differential abundance** (Mann-Whitney $U$ + Cohen's $d$) và **Enrichr**. Tuy vậy, em hoàn toàn đồng ý rằng việc bổ sung **Integrated Gradients / Input$\times$Gradient** (có dấu và chống bão hòa) kèm Adebayo randomization sanity checks sẽ làm phần diễn giải thuyết phục hơn rất nhiều."*

---

### Q11. [Double-Dipping (Post-Selection Inference) & FDR cục bộ trên tập 19–40 gen]
> **Hội đồng hỏi:**  
> *"Trong Mục 3.8 và 5.4, em huấn luyện mô hình trên 5 fold (tức mọi bệnh nhân đều xuất hiện trong tập train 4/5 lần), chọn ra 109 consensus features, rồi chạy **Mann-Whitney $U$ trên TOÀN BỘ 511 bệnh nhân** và chỉ tính **Benjamini-Hochberg FDR bên trong tập nhỏ 19–40 gen đó** (thay vì toàn bộ 20,000 gen)!*
> *Trong thống kê sinh học, đây là lỗi **double-dipping / post-selection inference** kinh điển: các đặc trưng đã được bộ phân lớp có giám sát chọn ra tất nhiên sẽ có xác suất cao khác biệt giữa các lớp trên chính bộ dữ liệu đó, và chia FDR cho 20 thay vì 20,000 làm $q$-value nhỏ đi hàng trăm lần!"*

- **Cách đỡ đòn:**
  - **Không cãi — khẳng định chính xác câu hỏi thống kê mà bước này trả lời:**
    *"Thưa thầy, nếu mục tiêu là phát hiện biomarker mới (de novo discovery) thì đúng là không được dùng dữ liệu huấn luyện để kiểm định lại và bắt buộc phải hiệu chỉnh FDR toàn hệ gen trên một tập độc lập (independent validation cohort).*
    *Trong luận văn (Mục 3.8 và Mục 4.4), em định nghĩa rất chặt vai trò của bước này chỉ là **kiểm tra tính nhất quán hậu kiểm (post-hoc consistency check)** cho bộ phân lớp: với một đặc trưng mà mô hình phụ thuộc mạnh (`|∂logit/∂x|` cao, vốn **không có dấu**), liệu nó có thực sự chênh lệch đơn biến theo đúng chiều sinh học đã biết giữa phân nhóm đó và phần còn lại hay không (để tách biệt 61 đặc trưng có hiệu ứng rõ ràng $|d|\ge 0.3$ khỏi 48 đặc trưng 'gradient-only' — ví dụ cả 13 đặc trưng CNV ở Basal-like đều không qua lọc và được đánh dấu tròn rỗng ở Hình 5.2).*
    *Hơn nữa, các giá trị $|d|$ của các chương trình cốt lõi cực kỳ lớn ($|d| = 1.25$ đến $3.53$, ví dụ *FOXC1* $+3.53$, *HER2_pY1248* $+2.61$, *NDC80* $-2.04$) nên ngay cả dưới hiệu chỉnh toàn hệ gen (genome-wide Bonferroni/BH trên 20,000 gen), chúng vẫn có ý nghĩa thống kê tuyệt đối. Còn với các đề cử ngoài chương trình kinh điển như *TBC1D31*, em khẳng định rõ chỉ là **unconfirmed nomination**, cần kiểm chứng trên cohort ngoài như METABRIC."*

---

### Q12. [Bắt bẻ Table 5.6: Tại sao Enrichr Luminal A ra *Staphylococcus aureus infection* và 0 MSigDB Hallmark?]
> **Hội đồng hỏi:**  
> *"Mở Table 5.6 (Enrichr trên danh sách consensus Luminal A — 22 gen):*
> 1. *Tại sao một nghiên cứu ung thư vú lại ra term **KEGG: Staphylococcus aureus infection** ($p=0.034$) và **Chromosome Organization** chỉ dựa trên 2 gen `RNASE7, PXDNL`?*
> 2. *Tại sao **MSigDB Hallmark 2020** (vốn có các hallmark kinh điển nhất của Luminal A là *Estrogen Response Early/Late*, *E2F Targets*, *G2M Checkpoint*) lại trả về **ZERO** term đạt FDR < 0.05?"*

- **Tại sao chuyện này xảy ra (phải hiểu rõ bản chất tin sinh học để trả lời cực mượt!):**
  - *Staphylococcus aureus infection* trong KEGG có chứa các gen keratin biểu mô (`KRT14`, `KRT17` — vì độc tố/bám dính tụ cầu liên quan đến keratin biểu mô da). Khi chạy Over-Representation Analysis (ORA) trên danh sách rất ngắn (**chỉ 22 gen**), chỉ cần trùng 2/95 gen là Fisher exact test đã cho $p_{\text{adj}} = 0.034$!
  - Tại sao MSigDB Hallmark ra 0 term? Vì 2 lý do kỹ thuật của ORA:
    1. **Trộn lẫn gen tăng và gen giảm** (vì vanilla gradient lấy trị tuyệt đối $|\partial \text{logit}/\partial x|$ không có dấu!): danh sách 22 gen Luminal A trộn lẫn các gen keratin/ER *tăng* với các gen nguyên phân (`NDC80`, `GTSE1`, `DEPDC1B`) *giảm* — chia nhỏ số overlap vào từng Hallmark gene set.
    2. **Cỡ danh sách quá bé (22 gen từ ngưỡng cứng top-20 consensus 3/5 fold):** MSigDB Hallmark có kích thước tập gen chuẩn ~200 gen/hallmark; với query chỉ 22 gen trong đó có cả gen phân tách từ 3 lớp khác nhau (mRNA, CNV, RPPA quy đổi tên gen), xác suất giao cắt ở một tập Hallmark đơn lẻ thường chỉ 2–3 gen, không đủ lực thống kê cho Fisher exact test (ORA), trái ngược với **GSEA (Gene Set Enrichment Analysis)** chạy trên toàn bộ danh sách gen được xếp hạng có dấu!
- **Cách đỡ đòn:**
  - *"Thưa thầy, đó là đặc trưng rất quen thuộc của phương pháp ORA (Enrichr) trên danh sách gen ngắn và không phân dấu (22 gen):*
    - *Term `Staphylococcus aureus infection` là một **database annotation artifact** do `KRT14` và `KRT17` được gán chéo trong thư viện KEGG vì vai trò keratin biểu mô (cũng chính 2 gen này nằm trong `Estrogen signaling pathway` ở ngay dòng dưới!). Em giữ nguyên toàn bộ 5 term đạt ngưỡng FDR < 0.05 trong Table 5.6 để tránh chọn lọc có lợi (cherry-picking).*
    - *MSigDB Hallmark không có term < 0.05 vì ngưỡng lọc 3/5 fold giữ lại chỉ 22 gen và trộn chung cả gen tăng lẫn gen giảm do gradient lấy trị tuyệt đối; nếu dùng **ranked GSEA** trên toàn bộ bảng xếp hạng có dấu (hoặc chạy ORA tách riêng nhóm $d > 0$ và $d < 0$ với top-100 gen) sẽ thu hồi rõ các Hallmark E2F/G2M và Estrogen Response."*

---

### Q13. [Kiểm tra kiến thức Sinh học về *TBC1D31* — Phát hiện cực đắt giá từ Literature 2024!]
> **Hội đồng hỏi:**  
> *"Ở Mục 5.5, em nêu **TBC1D31** như một gen nổi bật phân tách Luminal A ($d \approx -1.39$) và Luminal B ($d \approx +0.91$). Em dẫn bài Lupi et al. (2024) và viết rằng bài đó 'notably links TBC1D7 to glycolysis in TNBC'. Em đã đọc kỹ bài Lupi et al. (2024) chưa? Và em có biết **TBC1D31 nằm ở locus nhiễm sắc thể nào** không — liệu nó có phải chỉ là gen hành khách (passenger gene) đồng khuếch đại với một oncogene kinh điển trong Luminal B?"*

- **Bằng chứng khoa học tìm được (CỰC GIÁ TRỊ — biến điểm yếu thành cú ghi điểm tuyệt đối!):**
  1. **Trong chính Lupi et al. (*Cell Death & Disease* 2024, 15:647, [DOI: 10.1038/s41419-024-07037-2](https://doi.org/10.1038/s41419-024-07037-2)):** Tác giả thực sự đã chọn **3 hit hàng đầu là `TBC1D31`, `TBC1D22B`, và `TBC1D7`** và chứng minh thực nghiệm rằng biểu hiện cao của **chính `TBC1D31`** có tính nhân quả trong việc thúc đẩy kiểu hình đường phân (glycolytic phenotype) ở dòng tế bào ung thư vú TNBC! (Báo cáo Chương 5 chỉ nhắc đến `TBC1D7` là hơi thiếu chính xác và tự làm yếu bằng chứng của chính mình!).
  2. **Vị trí nhiễm sắc thể:** `TBC1D31` nằm tại **8q24.13** — sát cạnh oncogene **`MYC` tại 8q24.21**! Khuếch đại vùng **8q24** là biến đổi CNV đặc trưng của **Luminal B** (tăng sinh cao) so với **Luminal A** (ổn định bộ gen hơn). Ngoài ra, nghiên cứu trên *Advanced Science* (2024, [PMC11516053](https://pmc.ncbi.nlm.nih.gov/articles/PMC11516053/)) chứng minh khuếch đại 8q24.13 làm tăng biểu hiện `TBC1D31` (hiệu ứng cis mạnh) và `TBC1D31` hoạt động như oncogene thật sự thông qua trục **TBC1D31–Rab22A–EGFR** (giảm thoái hóa EGFR) ở ung thư gan HCC!
- **Cách đỡ đòn (Sẽ khiến giáo sư Tin sinh học trầm trồ!):**
  - *"Thưa thầy, rất cảm ơn thầy nhắc đến điểm này:*
    - *Thứ nhất, đúng là trong Lupi et al. (2024), chính **`TBC1D31`** (cùng `TBC1D22B` và `TBC1D7`) nằm trong top-3 gen được xác nhận thúc đẩy chuyển hóa glycolysis ở dòng tế bào ung thư vú TNBC — em sẽ diễn đạt chính xác hơn chỗ trích dẫn này.*
    - *Thứ hai, về vị trí hệ gen, `TBC1D31` nằm ở **8q24.13**, rất gần **`MYC` (8q24.21)** — một vùng khuếch đại điển hình phân biệt Luminal B tăng sinh cao với Luminal A. Do đó, hoàn toàn có khả năng tín hiệu mRNA của `TBC1D31` ($d = -1.39$ ở LumA, $+0.91$ ở LumB) phản ánh phần nào hiệu ứng **cis-amplification 8q24** đồng hành với MYC, bên cạnh vai trò chức năng độc lập của nó trong điều hòa ciliogenesis (Senatore et al., EMBO J 2021) và tín hiệu EGFR/glycolysis (Advanced Science 2024; Cell Death & Disease 2024). Chính vì không thể tách biệt vai trò driver vs passenger chỉ từ độ nhạy gradient, em giữ lập trường rất chặt là chỉ xem nó như **unconfirmed nomination** cần kiểm chứng thực nghiệm."*

---

## NHÓM D: PHẢN BIỆN TÍNH MỚI SO VỚI LITERATURE 2024–2026

### Q14. [CHÍ MẠNG — Tính mới so với `CrossAttOmics` (2025) & `DoGMA` (2026)]
> **Hội đồng hỏi:**  
> *"Trong chính file `reference.bib` của em có 2 bài báo cực kỳ sát đề tài nhưng em KHÔNG HỀ thảo luận trong Chương 2:*
> 1. **`CrossAttOmics` (Beaude et al., *Bioinformatics* 2025, [DOI: 10.1093/bioinformatics/btaf302](https://doi.org/10.1093/bioinformatics/btaf302)):** dùng encoder từng omic + **directed cross-attention theo đồ thị liên kết điều hòa giữa các omics** (CNV $\to$ mRNA, mRNA $\to$ proteomics) — thậm chí trong bảng ghi chú nội bộ của em nó còn đạt **97.49% F1 trên LUNG**!*
> 2. **`DoGMA` (Ling et al., arXiv 2026, [arXiv:2608.08148](https://arxiv.org/html/2608.08148)):** một foundation model Transformer-MoE sử dụng **Central-Dogma Attention** định hướng DNA $\to$ RNA $\to$ Protein trên TCGA và chuyển giao sang METABRIC!*
> *Vậy đóng góp mới (novelty) của luận văn em nằm ở đâu so với hai công trình này?"*

- **So sánh kỹ thuật chính xác (nắm lòng bảng này để trả lời không vấp 1 chữ!):**

| Tiêu chí | `CrossAttOmics` (Bioinformatics 2025) | `DoGMA` (arXiv 2026) | **Đề xuất trong Luận văn** |
|---|---|---|---|
| **Đơn vị Token** | Nhóm đặc trưng (feature-group decomposition) + intra-omic self-attention | 1 token / modality (6 modalities) trong Transformer-MoE | **Sequence of $T$ factorized latent tokens** (`OmicToTokens`) giải quyết suy biến softmax $T=1$ |
| **Cơ chế định hướng** | Cứng theo đồ thị tương tác cặp nguồn–đích (`source -> target`), không có gated residual mixing | **Soft bias** trong logit attention ($B_{\text{dogma}} = A + R_{ij}$) + pretraining tái tạo phân cấp quy mô lớn | **Directed multi-head cross-attention + Token-wise Gated Residual Fusion** ($F_i = Z_i + \alpha_i \odot \Delta_i$) + 2-layer Transformer `[CLS]` |
| **Kiểm chứng hướng (Direction Test)** | So đồ thị sinh học với đồ thị đầy đủ trên bài toán phân loại 18 loại ung thư | Ablation soft bias trong pretraining pan-cancer | **Kiểm chứng đối chứng đủ cả 6/6 hoán vị** dưới giao thức nested 5-fold khóa chặt + **ablation đối chứng trực tiếp $T=1$ vs $T>1$** (Table 5.5) |
| **Chế độ dữ liệu** | Pan-cancer 5,862 bệnh nhân (18 loại ung thư) | Pretraining pan-cancer foundation model | **Supervised small-sample subtyping** ($N = 222\text{–}682$, không cần pretraining ngoài) |

- **Cách đỡ đòn:**
  - *"Thưa thầy, đây là hai công trình cùng xu hướng đưa tri thức định hướng sinh học vào cơ chế attention mà em có theo dõi trong quá trình làm nghiên cứu. Điểm khác biệt và đóng góp cụ thể của luận văn em nằm ở 3 khía cạnh:*
    1. **Kiến trúc đại diện & Cơ chế kiểm soát residual:** Khác với `DoGMA` (dùng 1 vector/modality và phụ thuộc pretraining quy mô pan-cancer trên Transformer-MoE) và `CrossAttOmics` (dùng feature-group attention không có cơ chế gate thích nghi theo từng tọa độ token), mô hình của em kết hợp **factorized multi-token bottleneck (`OmicToTokens`)** với **token-wise gated fusion** sau cả cross-attention lẫn self-attention, tối ưu riêng cho bài toán phân phân nhóm cỡ mẫu nhỏ ($N=222\text{–}682$) huấn luyện từ đầu (from scratch).
    2. **Phát hiện thực nghiệm về tương tác giữa số lượng token và cơ chế attention (Table 5.4 & 5.5):** Luận văn chứng minh trực tiếp hiện tượng đổi dấu (sign flip): ở chế độ $T=1$ (single-token), bỏ self-attention lại *cải thiện* F1 (+0.94%), trong khi ở chế độ $T=256$ (multi-token), bỏ self-attention làm *giảm* F1 (-1.38%). Đây là kết quả cơ chế (mechanistic ablation) rõ ràng về mặt kiến trúc.
    3. **Kiểm chứng phản chứng đầy đủ 6/6 hoán vị hướng:** Thay vì mặc định thứ tự Central Dogma luôn thắng, luận văn báo cáo trung thực toàn bộ 6 hoán vị trên 3 cohort, chỉ rõ rằng lợi ích của riêng thứ tự sinh học trên chế độ supervised cỡ mẫu nhỏ là khá khiêm tốn và nằm trong dao động fold — một phát hiện có giá trị cảnh báo giúp định hướng sang cơ chế học hướng linh hoạt (như soft bias của `DoGMA`)."*

---

### Q15. [So với `Pathformer` (2024) và `MOT` (2023) — Tại sao không dùng Biological Pathway Tokens?]
> **Hội đồng hỏi:**  
> *"`Pathformer` (Liu et al., *Bioinformatics* 2024, [PMC11139513](https://pmc.ncbi.nlm.nih.gov/articles/PMC11139513/)) ánh xạ trực tiếp đa omic thành 2,289 **biological pathway tokens** nên bản thân attention weights đã giải thích trực tiếp được pathway crosstalk mà không cần input gradient! Còn `MOT` (Osseni et al., *Bioinformatics* 2023) cũng là Multi-Omics Transformer được em nhắc ở Chương 2 nhưng lại không có mặt trong Table 5.1. Tại sao em lại dùng latent tokens vô danh thay vì pathway tokens, và tại sao không đưa `MOT` hay `unrestricted all-to-all cross-attention` vào Table 5.1?"*

- **Cách đỡ đòn:**
  - **Latent tokens (`OmicToTokens`) vs Biological Pathway Tokens (`Pathformer`):**
    - *Pathformer* dựa chặt vào cơ sở dữ liệu pathway cố định (KEGG/Reactome/PID/BioCarta): (i) bỏ sót các gen chưa được chú giải vào pathway (ví dụ nhiều gen mới như `TBC1D31` hay các biến thể CNV ngoại biên không nằm trong 2,289 pathway chuẩn), (ii) đòi hỏi bước chuẩn hóa toàn bộ mọi assay về cùng cấp gen (gene-level summary statistics) trước khi gộp (early fusion), rất khó xử lý tự nhiên các **phospho-protein đặc hiệu vị trí** trong RPPA (như `HER2_pY1248`, `ERALPHA_pS118`, `SHC_pY317`, `P27_pT198` — vốn mang trạng thái hoạt hóa khác hẳn mức protein tổng!).
    - Ngược lại, `OmicToTokens` là **data-driven và assay-agnostic**: bảo toàn nguyên vẹn mọi phospho-protein RPPA và toàn bộ 20,000+ gen mà không phụ thuộc tri thức pathway có sẵn, đổi lại phải dùng input gradient tách riêng để truy vết mức gen.
  - **Về `MOT` và `Unrestricted Cross-Attention`:**
    - Thừa nhận thẳng: trong Chương 2 và Chương 6, em đã chủ động ghi rõ **so sánh trực tiếp với unrestricted all-to-all cross-attention** và mở rộng thêm các peer Transformer như `MOT` / `CrossAttOmics` dưới cùng giao thức matched là hạn chế hiện tại và là ưu tiên số 1 ở hướng phát triển (Chapter 6, mục ii). Trong 5 baseline đã chạy của Table 5.1, em tập trung đại diện cân bằng giữa 2 họ lớn là **Graph fusion** (MOGONET, MoGCN, DeepMoIC) và **Attention/Gated fusion** (moBRCA-net, MoXGATE).

---

## NHÓM E: KHẢ NĂNG TỔNG QUÁT HÓA & TRIỂN KHAI LÂM SÀNG

### Q16. [External Validation, Missing Omics, & Thiếu Confusion Matrix]
> **Hội đồng hỏi:**  
> *"Ba câu hỏi thực tế lâm sàng:*
> 1. *Cả 3 cohort đều lấy từ **TCGA** (cùng pipeline Xena/TCPA). Không có external validation (như **METABRIC** cho BRCA hay **CPTAC** cho LUNG/COADREAD), làm sao biết mô hình 16.42M tham số không học batch effect của TCGA?*
> 2. *Trên thực tế bệnh viện, rất hiếm bệnh nhân có đủ cả CNV + mRNA + RPPA (bằng chứng: chính trong TCGA em phải loại bỏ một nửa bệnh nhân để lấy giao $N=511$ BRCA và $N=222$ COADREAD!). Mô hình của em yêu cầu bắt buộc đủ cả 3 assay — vậy ứng dụng lâm sàng kiểu gì?*
> 3. *Ở Mục 5.8 em suy đoán lỗi trên BRCA rơi vào biên Luminal A $\leftrightarrow$ Luminal B và trên COADREAD rơi vào CMS1/CMS3, nhưng lại **không in confusion matrix**!"*

- **Cách đỡ đòn:**
  1. **Về External Validation (METABRIC / CPTAC):** Thừa nhận rõ đây là giới hạn phạm vi của đồ án tốt nghiệp (đã ghi minh bạch ở Chương 5 dòng 267 và Chương 6 dòng 14). Giải thích thêm một khó khăn thực tế về dữ liệu: **METABRIC** có CNV + mRNA nhưng **không có RPPA proteomics** khớp chuẩn TCPA; trong khi **CPTAC** dùng **mass-spectrometry proteomics** (hàng nghìn protein, thang đo khác hoàn toàn với ~200 kháng thể RPPA). Do đó để đánh giá external thực sự, cần mở rộng kiến trúc hỗ trợ missing modality hoặc chuyển đổi không gian đặc trưng proteomics.
  2. **Về Missing Omics (đây là điểm mạnh ẩn của kiến trúc Token!):**
     - *"Tuy báo cáo hiện tại mới đánh giá chế độ đầy đủ 3 assay (matched complete-case), nhưng **kiến trúc token-level + token-wise gate của em có lợi thế tự nhiên để mở rộng sang missing omics** hơn hẳn concatenation hay graph:*
       - *Nếu thiếu CNV: đặt gate cross-attention mRNA $\alpha_T = 0$ (giữ nguyên $Z_T$) và bỏ chuỗi token $Z_G$ khỏi đầu vào Transformer (vốn hỗ trợ độ dài chuỗi biến thiên).*
       - *Nếu thiếu RPPA: bỏ bước 2 và chỉ đưa $[u_0; F_T; Z_G]$ vào Transformer.*
       - *Kết hợp với **modality dropout** lúc huấn luyện sẽ cho phép mô hình dự đoán linh hoạt khi thiếu 1–2 assay mà không cần imputation thô."*
  3. **Về Confusion Matrix:** Thừa nhận thiếu sót trực quan trong bản nộp (do runner chỉ log tổng hợp F1/Acc từng fold mà chưa xuất ma trận nhầm lẫn tổng hợp), giải thích cơ sở định lượng cho suy luận từ khoảng cách Accuracy–Macro-F1 (1.46% trên BRCA và 1.74% trên COADREAD), và xác nhận đây là việc đầu tiên cần bổ sung khi hoàn thiện bản công bố (Chapter 6, mục iv).

---

## CHECKLIST CÁC LỖI CHI TIẾT PHẢI NHỚ TRƯỚC KHI BƯỚC VÀO PHÒNG BẢO VỆ

1. **Tuyệt đối không dùng số liệu từ bộ slide cũ (`OSCAR` vector-level):**
   - Số liệu **CHUẨN** theo luận văn (`main.tex`):
     - Kiến trúc: `OmicToTokens` ($T=256, M=32, d_{\text{mid}}=256$), 4 heads, 769 tokens (`[CLS]` + $3 \times 256$), **16,420,996 tham số** trên BRCA.
     - Kết quả chính (Table 5.1): BRCA **89.93 ± 2.15% F1** (91.39% Acc), LUNG **97.34 ± 1.30% F1** (97.36% Acc), COADREAD **89.26 ± 2.55% F1** (91.00% Acc), **Avg F1 = 92.18%** (Avg Acc = 93.25%).
     - Ablation (Table 5.2): `Concatenation` = **91.44% F1**; `Without self-attn` = **90.80% F1** (-1.38); `Without cross-attn` = **91.38% F1** (-0.80).
     - Ordering (Table 5.3): Biological **92.18% F1** (#1 Avg, #1 LUNG), Reverse **92.11% F1** (#1 BRCA: 90.85%), Mixed **91.91% F1** (#1 COADREAD: 90.05%).
     - Token Grid (Table 5.4–5.5): $1\times 128$ = **90.75% F1** (bỏ self-attn tăng lên 91.69%); $128\times 32$ = **92.41% F1** (13.25M params).
     - BRCA Saliency: **109 consensus features**, **61/109 passing** ($q<0.05, |d|\ge 0.3$), 8 cặp CNV–mRNA đồng hướng (2 cặp qua lọc ở HER2: *GRB7*, *TCAP*), đề cử ngoài chương trình kinh điển: **`TBC1D31`** (không phải `GAS2L3`).
     - Gate BRCA: mRNA mean **0.502–0.504** (frac >0.5: 48.1–50.7%), Proteomics mean **0.513–0.526** (frac >0.5: 57.0–64.6%).
2. **Đính chính trích dẫn nếu bị hỏi:**
   - **CMS1–CMS4** trên COADREAD do **Guinney et al., *Nature Medicine* 2015** đề xuất (không phải TCGA 2012).
   - **Lupi et al., *Cell Death & Disease* 2024** nghiên cứu trực tiếp **`TBC1D31`** (cùng `TBC1D22B`, `TBC1D7`), và `TBC1D31` nằm tại **8q24.13** sát cạnh `MYC` (8q24.21).
3. **Nguyên tắc vàng 3 bước khi bị Hội đồng "vả":**
   - **Bước 1 — Ghi nhận & Đồng tình có kiểm soát:** *"Thưa thầy, nhận xét của thầy rất sắc / hoàn toàn chính xác về mặt toán học (hoặc thống kê)..."* — Tuyệt đối không phòng thủ cảm tính hay cãi cố.
   - **Bước 2 — Đưa ngay bằng chứng từ chính bảng/hình trong luận văn:** Trích dẫn đúng số Table/Figure (ví dụ Table 5.2, Table 5.5, Figure 5.4, Section 4.4).
   - **Bước 3 — Chốt ranh giới khoa học (Claim Boundary) & Hướng giải quyết cụ thể:** Chỉ rõ luận văn đã giữ tuyên bố ở mức nào (descriptive / consistency check) và nêu tên phương pháp kỹ thuật chuẩn xác để giải quyết (ví dụ: Nadeau–Bengio corrected t-test, soft bias như DoGMA, Integrated Gradients + sanity checks, modality dropout).
