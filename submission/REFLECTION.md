# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là run `attn_only` dù train loss thấp hơn hẳn `correct` (0.537 so với 0.626) nhưng khi đo trên target task thực tế thì không hề thắng, mà hoà nhau ở 0.9375. Tôi nhận ra train loss là một chỉ số thay thế (surrogate metric) dễ gây ngộ nhận nếu chỉ chăm chăm tối ưu nó mà không đo downstream task. Ngoài ra việc QLoRA 4-bit bị tụt gần 10% accuracy trên model 4B cũng là một bất ngờ lớn so với những gì thường được quảng cáo.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Phần mất nhiều thời gian nhất là lúc chạy pipeline đo đạc đánh giá các baseline (NB2) và chạy 3 run đối chứng (NB4) trên Colab vì việc sinh văn bản (text generation) cho cả tập dữ liệu lặp lại nhiều lần rất tốn thời gian. Ban đầu tôi nghĩ bước huấn luyện NB3 sẽ lâu nhất, nhưng thực tế việc sinh văn bản để đo đạc và đối chứng công bằng mới là phần chiếm thời gian áp đảo.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước đây tôi từng tin rằng: "Cứ fine-tune là model sẽ thông minh hơn toàn diện và rank càng to thì càng tốt". Giờ tôi hiểu rằng fine-tune trên dữ liệu chuyên biệt hẹp rất dễ bị hiện tượng Quên thảm hoạ (Catastrophic Forgetting) làm hỏng kiến thức tổng quát (như độ tụt regression -0.125 ở NB5), và vị trí đặt adapter (`all-linear`) quan trọng hơn nhiều so với việc chỉ nhồi rank vào các lớp Attention.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để giải thích luồng hoạt động của từng notebook, hỗ trợ phân tích các file kết quả JSON/CSV và viết báo cáo đánh giá logic. Điểm cần lưu ý là AI ban đầu có thể suy diễn kết luận mô hình thắng hay thua theo cảm tính nếu không ép đọc chính xác các con số thực nghiệm từ `runs.csv` và `verdict.json`.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tuyệt đối không phải là mở code train LoRA ngay, mà là: (1) Kiểm tra Chat Template và chứng minh Loss Mask chuẩn xác (chỉ tính loss trên câu trả lời, không tính prompt), (2) Đóng băng một bộ test đánh giá và đo trước một Baseline Prompting thật mạnh (Few-shot) để xem prompting đã giải quyết được bao nhiêu phần trăm bài toán trước khi quyết định tốn chi phí fine-tune.

