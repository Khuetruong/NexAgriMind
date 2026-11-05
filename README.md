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
| 2. Kho tài liệu khoa học | `src/agrimind/corpus`, `data/sources` | `agrimind-fetch` tải tài liệu mở (web, PDF) theo danh sách chuyên gia chọn, tự tra giấy phép bài báo trên OpenAlex, dừng khi giấy phép mâu thuẫn; chưa có tài liệu thật |
| 3. Trợ lý hỏi đáp (RAG) | `src/agrimind/rag`, `src/agrimind/api` | BM25, thêm tìm theo nghĩa (Qdrant + bge-m3) gộp bằng RRF; Qwen3 qua vLLM; chạy thử được với `StubLLM` |
| 4. Đồ thị tri thức | `src/agrimind/kg`, `data/kg` | Cấu trúc + 3 đồ thị mẫu: cà phê năng suất giảm, gà đẻ giảm trứng, tôm giảm ăn (chưa chuyên gia duyệt); trợ lý dùng đồ thị để liệt kê các khả năng (bỏ tương quan); API `/v1/kg/...`; nạp Neo4j bằng `agrimind-kg` |
| Bảo vệ API | `src/agrimind/security`, `src/agrimind/audit` | Khoá riêng từng bên + quyền, giới hạn tần suất và kích thước, nhật ký không chứa nội dung |
| Trạm kiểm soát dữ liệu ra (A07) | `src/agrimind/gateway/egress.py` | Điểm trong nước chỉ gửi câu hỏi đã bỏ nhận dạng lên Core; còn sót thì chặn |
| 6. Đồng ý dữ liệu (phía AgriMind) | `src/agrimind/consent` | Điểm trong nước chỉ dùng dữ liệu trại khi trại đã bật trong Nexagri; hỏi Nexagri không được thì coi như chưa đồng ý |
| 5. Bộ chấm điểm | `src/agrimind/eval`, `benchmarks` | Chấm tự động 4 tiêu chí; `agrimind-review` xuất bảng cho chuyên gia chấm (Excel) và báo cáo cửa giai đoạn 1; bộ câu hỏi v0 là bản nháp |
| 7. Hạ tầng | `deploy/` | Cấu hình chạy thật cho điểm Việt Nam và Core: HTTPS, mạng nội bộ, mật khẩu Qdrant/Neo4j, vLLM, sao lưu. Xem `deploy/README.md` |

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

Tìm theo nghĩa: chạy Qdrant (`docker compose --profile vector up qdrant`) và một dịch vụ embedding tương thích OpenAI chạy `BAAI/bge-m3`, rồi:

```bash
export AGRIMIND_QDRANT_URL=http://localhost:6333 AGRIMIND_EMBED_URL=http://<máy>:8080/v1
agrimind-index tests/fixtures/corpus     # chỉ nhúng tài liệu có giấy phép thương mại
```

Qdrant hoặc embedding hỏng thì trợ lý tự quay về BM25, không ngừng dịch vụ.

Đồ thị tri thức vào Neo4j (`docker compose --profile graph up neo4j`):

```bash
AGRIMIND_NEO4J_PASSWORD=agrimind-dev agrimind-kg data/kg/*.yaml --neo4j-url http://localhost:7474
```

Nối Nexagri (chỉ điểm `country`): đặt `NEXAGRI_TENANT_URL` (dịch vụ tenant của Nexagri) và `INTERNAL_SERVICE_KEY` (cùng khoá với Nexagri). Dữ liệu trại gửi kèm câu hỏi chỉ được dùng khi trại đã bật chia sẻ. Nexagri có thể báo bật/tắt ngay qua `POST /v1/internal/consent-events`.

## Cho bên ngoài truy cập API

Mọi đường `/v1/...` cần khoá: `Authorization: Bearer amk_<mã bên>.<chuỗi bí mật>`. Chỉ `/health` mở.

1. Tạo khoá: `agrimind-keys new nexagri-vn --name "Nexagri Việt Nam" --scopes ask:general,ask:farm,consent:events --countries VN --rate 300`.
   Khoá chỉ in ra một lần; giao cho bên kia qua kênh an toàn. Đoạn YAML in ra chỉ có mã băm, dán vào tệp `AGRIMIND_CLIENTS_FILE`.
2. Quyền: `ask:general` (câu hỏi chung, cấp được cho đối tác), `ask:farm` (kèm dữ liệu trại, chỉ Nexagri trong nước), `consent:events` (báo bật/tắt chia sẻ, chỉ Nexagri).
3. Thu hồi: đặt `disabled: true` hoặc `expires:` cho bên đó rồi khởi động lại.
4. Production (`AGRIMIND_ENV=production`): không có tệp khoá thì dịch vụ không chạy; trang `/docs` tắt. Vẫn đặt HTTPS và giới hạn tần suất ở cổng phía trước (nginx).
5. Nhật ký `AGRIMIND_AUDIT_FILE` ghi ai gọi, kết quả, trại nào bị dùng dữ liệu, cái gì đi ra Core; không ghi câu hỏi hay dữ liệu trại.

Điểm trong nước hỏi Core khi kho trong nước chưa đủ: đặt `AGRIMIND_CORE_URL` và `AGRIMIND_CORE_KEY` (khoá do Core cấp, chỉ quyền `ask:general`).

## Việc còn lại cần người (không làm được bằng mã)

- Chuyên gia chọn nguồn (`data/sources`), duyệt đồ thị, sửa bộ câu hỏi và chấm câu trả lời (gói 5, 9).
- Thuê máy chủ, tên miền (gói 7); luật sư xem ba luồng dữ liệu (gói 8); trang trại thí điểm bật đồng ý (gói 9).
- Giai đoạn 2 trở đi (mô hình riêng cho từng trung tâm, Vision AI, dự báo) cần dữ liệu thật từ các việc trên.

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
