# Lab 21 — Evaluation Report

**Họ tên**: Vũ Minh Trí  **MSSV**: 2A202602629 **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Google Colab)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.
>
> **Mẫu này là gợi ý.** Bạn được tự chọn base model, dataset và tự viết report theo cấu
> trúc của mình — miễn là có đủ: lựa chọn + lý do, bằng chứng mask, mốc đóng băng, kết quả,
> phán quyết, điều học được (rubric 4.1).

---

## 1. Setup

| | |
|---|---|
| Dataset | Ticket CSKH tiếng Việt → JSON triage 4 trường (250 mẫu) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 (tier T4) — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs (khoảng 30 steps với effective batch size = 16) |

**Template có giữ khối `<think>` không?** **Có** — Chat template giữ nguyên khối suy luận (`reasoning preserved — safe to train on traces` theo *results/template_check.json*).
Nếu không: Không cần can thiệp vì template gốc đã hỗ trợ bảo toàn thẻ suy luận.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3180.3 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1003.2 |
| (c) LoRA fine-tune | 0.970 | 0.456 | 1.000 | 1398.6 |

**(b) có thật sự mạnh hơn (a) không?** **Có**. Prompt tối ưu (b) cải thiện vượt bậc trên toàn bộ 50 mẫu test: điểm target tăng từ 0.000 lên 0.765, format đạt tuyệt đối 1.000 (100% tuân thủ cấu trúc JSON 4 key), và độ trễ giảm hơn 3 lần (1003.2 ms so với 3180.3 ms của naive prompt do naive prompt sinh văn bản lan man không cấu trúc).
Bạn có sửa `OPTIMIZED_PROMPT` không? **Không sửa**. Prompt (b) được giữ nguyên vẹn đúng theo file gốc của bài lab (SHA: `719e74d3b6232053`) để đảm bảo tính khách quan và liêm chính học thuật, làm thước đo chuẩn mực trước khi huấn luyện mô hình.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6263 | **0.9700** | 378.8 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5372 | **0.9700** | 252.8 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.0000** | 375.2 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.9400** | 445.2 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

Trên toàn bộ 50 mẫu target, `attn_only` và `correct` đạt kết quả **hoà nhau** (cùng đạt độ chính xác ấn tượng 0.9700), mặc dù `attn_only` train nhanh hơn (252.8s so với 378.8s). Thứ tự này **hoàn toàn trái ngược với thứ tự theo train loss**: theo train loss thì `attn_only` (0.5372) trông "ngon hơn" `correct` (0.6263). Điều này chứng minh rằng việc dồn toàn bộ tham số vào rank cực lớn ($r=283$) chỉ ở 2 module $q, v$ dễ gây overfitting trên tập huấn luyện nhưng không đem lại ưu thế vượt trội trên downstream task so với việc phân bổ đều rank vừa phải ($r=16$) trên toàn bộ 12 khối tuyến tính (`text-linear`). Vị trí bao phủ adapter trên các lớp MLP và Attention là nhân tố quyết định độ tổng quát hóa vững chắc hơn việc đơn thuần tăng rank.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Run `wrong_lr` chỉ khác duy nhất learning rate ($10^{-5}$ thay vì $10^{-4}$), và loss kết thúc ở mức 1.5702, cao gấp hơn 2.5 lần so với `correct`. Nếu chỉ nhìn vào đường loss này mà không biết thông tin về LR, người phát triển sẽ dễ dàng kết luận sai lầm rằng dữ liệu huấn luyện quá ít, kiến trúc LoRA không đủ năng lực học hoặc bài toán phân loại JSON 4 trường quá phức tạp đối với model 4B. Trong thực tế, vì LoRA chỉ cập nhật một ma trận nhân rank thấp nên bắt buộc cần learning rate lớn hơn 10x so với Full Fine-Tuning để đưa các tham số di chuyển đủ xa; mức LR $10^{-5}$ khiến adapter hầu như dậm chân tại chỗ và sụp đổ hoàn toàn ở khâu đánh giá (target = 0.000, format = 0.000).

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

Run `qlora` 4-bit tiết kiệm lượng VRAM vô cùng ấn tượng: đỉnh bộ nhớ giảm từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm tới 56% VRAM). Tuy nhiên, cái giá phải trả rất đáng kể: thời gian huấn luyện tăng lên 445.2s (chậm hơn 17.5% so với 16-bit LoRA do overhead lượng tử hoá/giải lượng tử hoá liên tục), đồng thời độ chính xác target tụt từ 0.9700 xuống 0.9400 (suy giảm 3.0% độ chính xác). Con số đo đạc thực nghiệm này hoàn toàn ủng hộ khuyến nghị chính thức của nhóm tác giả Qwen: không nên dùng QLoRA 4-bit cho họ mô hình này nếu tài nguyên VRAM (như card T4 16GB) đã đủ sức chạy bf16/fp16 LoRA nguyên bản.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.336` · `valid_trace_rate = 0.00`

Diễn giải (≥100 từ): 
Cổng hồi quy đánh giá phán quyết là `FAILED` không phải vì mô hình fine-tune kém về mặt chuyên môn (thực tế điểm target đạt tới 0.970 trên 50 mẫu, tăng mạnh $\Delta = +0.205$ hay +20.5% so với prompt tối ưu b). Nguyên nhân thất bại duy nhất là vi phạm ngưỡng an toàn năng lực tổng quát: điểm `regression` bị tụt mạnh $-0.336$ (từ 0.791 xuống 0.456, trong khi dung sai cho phép chỉ là $-0.020$). Đây là minh chứng kinh điển cho hiện tượng "Quên thảm hoạ" (Catastrophic Forgetting) khi fine-tune LLM trên một tập dữ liệu chuyên biệt hẹp (225 ticket CSKH dạng JSON) mà hoàn toàn thiếu vắng các mẫu dữ liệu tri thức thông thường. Mô hình đã bị co cụm không gian biểu diễn vào cấu trúc JSON, dẫn tới suy giảm nghiêm trọng khả năng trả lời các câu hỏi kiến thức nền tảng. Kết quả này phản ánh chính xác thực tế kỹ thuật và chỉ ra giải pháp cần thiết trước khi deploy: trộn 1–5% replay data (tri thức đa lĩnh vực) vào pipeline huấn luyện để bảo toàn năng lực tổng quát.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | Lệch urgency/sentiment | Đúng hoàn toàn 4 trường JSON | ✅ FT thắng |
| 2 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | Đoán sai intent | Đúng hoàn toàn 4 trường JSON | ✅ FT thắng |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | Đoán đúng urgency: `thap` | Đoán sai urgency: `trung_binh` | ❌ **FT thua** |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. | `san_pham_loi`, `thap`, `nồi chiên không dầu`, `trung_tinh` | Đoán đúng urgency: `thap` | Đoán sai urgency: `trung_binh` | ❌ **FT thua** |
| 5 | Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình. | `doi_tra`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | Format JSON chập chờn | Đúng hoàn toàn 4 trường JSON | ✅ FT thắng |

Có mẫu chung nào ở các ca FT thua không?
Có một pattern rất rõ ràng ở cả 2 ca FT thua (ca #3 và ca #4): Mô hình fine-tune đều đoán sai trường `urgency` từ mức nhãn đúng là `"thap"` thành `"trung_binh"`. Trong tập dữ liệu huấn luyện, mức độ ưu tiên `"trung_binh"` chiếm tỷ lệ áp đảo so với `"thap"`. Khi gặp các từ khoá diễn đạt sự kiên nhẫn nhẹ nhàng như *"Khi nào tiện"*, mô hình fine-tune có xu hướng thiên lệch (inductive bias) về lớp nhãn phổ biến nhất (`trung_binh`), trong khi mô hình prompt tối ưu (b) lại suy luận ngữ nghĩa tinh tế hơn dựa trên ngữ cảnh vài mẫu few-shot.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** 
Dựa trên kết quả đo lường bốn nhóm chỉ số độc lập trên toàn bộ 50 mẫu kiểm thử, kết luận trung thực là: **Chưa nên deploy bản fine-tune này lên production ngay lập tức ở trạng thái hiện tại**, bất chấp việc mô hình đạt độ chính xác tác vụ chuyên biệt cực kỳ ấn tượng (target đạt 0.970 so với 0.765 của prompt b, độ tuân thủ format đạt 100%). Lý do cốt lõi nằm ở sự sụt giảm nghiêm trọng năng lực tổng quát (regression sụt giảm $\Delta = -0.336$), vi phạm nghiêm trọng cổng an toàn của hệ thống. Trong một hệ thống chatbot CSKH thực tế, sự suy thoái này có thể khiến trợ lý ảo trở nên ngô nghê khi khách hàng hỏi các câu hỏi xã giao, tra cứu thông tin phổ thông ngoài luồng ticket. 

Đòn bẩy thật sự trong lab này không nằm ở việc cố gắng đẩy rank LoRA lên thật cao (bằng chứng là $r=283$ của `attn_only` không hề thắng được $r=16$ của `correct`), mà nằm ở **vị trí đặt adapter (`text-linear`)**, **learning rate chuẩn xác ($10^{-4}$ thay vì $10^{-5}$)**, và đặc biệt là **chất lượng cân bằng dữ liệu cùng loss mask**. Bản fine-tune chỉ sẵn sàng deploy sau khi được huấn luyện lại với 3–5% dữ liệu hội thoại đa năng (replay data) để chặn đứng hiện tượng catastrophic forgetting.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Không dùng surrogate metric để đánh giá:** Train loss thấp hơn không đồng nghĩa với model tốt hơn. Run `attn_only` có loss 0.537 (thấp hơn `correct` 0.626) nhưng điểm target thực tế chỉ ngang bằng và có nguy cơ ghi nhớ vẹt.
2. **Quy tắc đòn bẩy cấu hình LoRA:** Vị trí bao phủ toàn bộ các ma trận tuyến tính (`all-linear`) quan trọng hơn nhiều so với việc chỉ gắn vào Attention rồi bù đắp bằng rank cao; đồng thời LR của LoRA bắt buộc phải đặt ở thang $10^{-4}$ (gấp 10 lần full fine-tune).
3. **Phép đo ba baseline là bắt buộc trước khi train:** Đóng băng baseline (b) có prompt tối ưu trước khi huấn luyện giúp ngăn chặn thiên kiến tự lừa dối bản thân (viết prompt yếu đi để tâng bốc bản fine-tune), và chỉ số regression là chiếc phanh an toàn không thể thiếu để phát hiện hiện tượng quên thảm hoạ.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
1. Trộn thêm 3% dữ liệu hội thoại tổng quát tiếng Việt vào tập train để khắc phục lỗi hồi quy regression, đưa kết quả cổng kiểm tra về trạng thái `PASSED`.
2. Thực hiện sweep learning rate tinh chỉnh quanh $1.5 \times 10^{-4}$ để xem liệu có thể đẩy target lên 1.000 tuyệt đối hay không.
3. Thực hiện merge LoRA adapter vào base model bằng script NB6 và kiểm tra độ trễ inference khi phục vụ (serving).

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

