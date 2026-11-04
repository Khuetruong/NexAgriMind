# NexAgriMind – AgriMind Agricultural Intelligence OS

Mã nguồn của AgriMind, hạ tầng AI nông nghiệp dùng chung mà Nexagri và các đối tác kết nối vào.
Thiết kế theo bộ Master Blueprint A01–A10. Kế hoạch tổng: https://claude.ai/artifact/3jXono3E8xK912Pt8Dmu7L

## Nguyên tắc không được phá

1. **Dữ liệu người dùng ở lại từng nước.** AgriMind Core (dùng chung toàn cầu) không bao giờ nhận dữ liệu trang trại. Dịch vụ chạy với `AGRIMIND_ROLE=core` sẽ từ chối mọi yêu cầu có mã trang trại hoặc dữ liệu người dùng (A03, A07).
2. **Mọi câu trả lời có nguồn và có độ tin cậy**, theo mẫu 6 phần: kết luận, bằng chứng, độ tin cậy, giả định, cách kiểm chứng, việc cần chuyên gia xác nhận (A05). Không đủ nguồn thì từ chối. Nguồn mô hình tự bịa bị loại bằng mã.
3. **Tương quan không phải nhân quả.** Trong đồ thị tri thức, quan hệ `correlates_with` không được dùng khi tìm nguyên nhân (A02, A03).
4. **Chỉ dùng tài liệu và mô hình có giấy phép thương mại** cho sản phẩm (A04).
5. **AI đề xuất, con người quyết định.** Không tự cho thuốc, hóa chất hay điều khiển thiết bị (A06).

## Có gì trong giai đoạn 1 (tháng 1–3, A10)

| Gói | Thư mục | Trạng thái |
|---|---|---|
| 1. Khung dự án | toàn bộ, `.github/workflows/ci.yml` | Có |
| 2. Kho tài liệu khoa học | `src/agrimind/corpus` | Bộ nạp + kiểm tra giấy phép; chưa có tài liệu thật |
| 3. Trợ lý hỏi đáp (RAG) | `src/agrimind/rag`, `src/agrimind/api` | BM25 + Qwen3 qua vLLM (API tương thích OpenAI); chạy thử được với `StubLLM` |
| 4. Đồ thị tri thức | `src/agrimind/kg`, `data/kg` | Cấu trúc + đồ thị mẫu "tôm giảm ăn" (chưa chuyên gia duyệt) |
| 5. Bộ chấm điểm | `src/agrimind/eval`, `benchmarks` | Chấm tự động 4 tiêu chí; bộ câu hỏi v0 là bản nháp |

Tài liệu trong `tests/fixtures` là **mẫu kiểm thử**, không phải tri thức khoa học.

## Chạy thử

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q
agrimind-ingest tests/fixtures/corpus
agrimind-eval --corpus tests/fixtures/corpus --bench benchmarks/vi-core-v0.yaml
uvicorn agrimind.api.app:app --reload   # http://localhost:8000/docs
```

Nối mô hình thật: chạy vLLM với Qwen3 rồi đặt `AGRIMIND_LLM_URL=http://<máy>:8000/v1`, `AGRIMIND_LLM_MODEL=Qwen/Qwen3-8B`.

## Thêm tài liệu

Mỗi tài liệu là một file `.md` có phần đầu YAML. Bắt buộc có `id`, `title`, `source`, `license`:

```yaml
---
id: fao-xxxx-2024
title: "..."
source: "https://..."
license: CC-BY        # CC0 | CC-BY | CC-BY-SA | CC-BY-NC | public-domain | gov-public | partner | proprietary | unknown
domains: [D02, D04]   # mã trung tâm D01..D10
year: 2024
---
```

Tài liệu `CC-BY-NC`, `proprietary`, `unknown` vẫn nạp được nhưng bị loại khỏi kho phục vụ sản phẩm.
