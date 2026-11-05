# Chạy AgriMind thật (gói 7 của A10)

Hai cụm máy, đúng kiến trúc A07:

| Cụm | Thư mục | Đặt ở đâu | Giữ dữ liệu trang trại? |
|---|---|---|---|
| Điểm trong nước Việt Nam | `deploy/vn` | Máy chủ tại Việt Nam (cùng nơi với Nexagri) | Có, và chỉ ở đây |
| Core dùng chung | `deploy/core` | Đề xuất Singapore | Không bao giờ |

## Máy cần thuê (giai đoạn 1)

- **Điểm Việt Nam:** 1 máy ảo 4 vCPU, 8 GB RAM, 80 GB đĩa (ví dụ Viettel IDC T1.Base 05). Chưa cần GPU: đặt `AGRIMIND_LLM_URL` trỏ tới máy GPU thuê theo giờ khi chạy thử mô hình thật.
- **Core:** giai đoạn 1 có thể chưa cần. Khi cần thì dùng cấu hình giống điểm Việt Nam.
- **Có GPU** (từ tháng 4): chạy thêm `--profile gpu` để có vLLM với Qwen3-8B trên cùng máy, cần GPU ≥ 24 GB.

## Các bước dựng điểm Việt Nam

1. Cài Docker và Docker Compose. Trỏ tên miền (ví dụ `api-vn.<tên miền>`) về IP máy. Mở cổng 80 và 443, đóng các cổng khác.
2. Lấy mã nguồn: `git clone` kho này vào `/opt/nexagrimind`.
3. Cấu hình:
   ```bash
   cd /opt/nexagrimind/deploy/vn
   cp .env.example .env              # điền mật khẩu (openssl rand -base64 32), địa chỉ Nexagri
   cp clients.example.yaml clients.yaml
   pip install /opt/nexagrimind      # để có lệnh agrimind-keys
   agrimind-keys new nexagri-vn --name "Nexagri Việt Nam" \
     --scopes ask:general,ask:farm,consent:events --countries VN --rate 300
   # dán đoạn YAML vào clients.yaml; khoá in ra thì đặt vào AGRIMIND_API_KEY của Nexagri
   ```
4. Chạy: `docker compose up -d --build`. Kiểm tra: `curl https://api-vn.<tên miền>/health`.
5. Nạp tài liệu và đồ thị:
   ```bash
   docker compose cp ../../corpus/. agrimind:/var/lib/agrimind/corpus/   # tài liệu do agrimind-fetch tải
   docker compose restart agrimind
   docker compose exec agrimind agrimind-index /var/lib/agrimind/corpus \
     --qdrant-url http://qdrant:6333 --embed-url http://embed:80/v1
   docker compose exec -e AGRIMIND_NEO4J_PASSWORD=$NEO4J_PASSWORD agrimind \
     agrimind-kg /app/data/kg/*.yaml --neo4j-url http://neo4j:7474
   ```
   Khi `docker compose exec`, Qdrant cần khoá: thêm `-e AGRIMIND_QDRANT_API_KEY=$QDRANT_API_KEY`.
6. Sau khi mô hình nhúng (và vLLM nếu có) đã tải xong lần đầu, xoá `models_download` khỏi hai dịch vụ đó trong `docker-compose.yml` rồi `docker compose up -d`. Từ đó các dịch vụ xử lý dữ liệu không ra Internet được.
7. Sao lưu hằng ngày bằng cron: `deploy/backup.sh /opt/nexagrimind/deploy/vn /backup/agrimind`. Bản sao lưu phải lưu **trong nước**.

## Nối Nexagri

Trong cấu hình Nexagri (dịch vụ `ai` và `tenant`):

```
AGRIMIND_URL=https://api-vn.<tên miền>
AGRIMIND_API_KEY=<khoá agrimind-keys in ra>
AGRIMIND_COUNTRY=VN
```

Trên AgriMind, `NEXAGRI_TENANT_URL` trỏ tới dịch vụ tenant của Nexagri và `INTERNAL_SERVICE_KEY` trùng khoá nội bộ của Nexagri. Đường nối giữa hai bên nên đi qua mạng riêng (VPC) hoặc HTTPS.

## Kiểm tra an toàn trước khi mở cho đối tác

- [ ] `AGRIMIND_ENV=production` (đã đặt sẵn trong compose): thiếu `clients.yaml` thì dịch vụ không chạy, `/docs` bị tắt.
- [ ] Gọi `/v1/ask` không có khoá phải bị trả 401.
- [ ] Khoá cấp cho đối tác chỉ có `ask:general`.
- [ ] Qdrant, Neo4j, mô hình không có cổng công khai (`docker compose ps` chỉ thấy Caddy mở 80/443).
- [ ] `.env` và `clients.yaml` không nằm trong git.
- [ ] Nhật ký `audit.jsonl` được sao lưu và không chứa nội dung câu hỏi.
