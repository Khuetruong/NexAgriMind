#!/usr/bin/env bash
# Sao lưu hằng ngày cho một điểm AgriMind (chạy bằng cron trên máy chủ).
#   0 2 * * *  /opt/nexagrimind/deploy/backup.sh /opt/nexagrimind/deploy/vn /backup/agrimind
# Lưu: kho tài liệu, nhật ký kiểm toán, danh sách khoá (chỉ mã băm), bản chụp Qdrant.
# Đồ thị tri thức nằm trong git (data/kg) nên không cần sao lưu Neo4j; nạp lại bằng agrimind-kg.
# Bản sao lưu của điểm trong nước PHẢI lưu trong nước (A07).
set -euo pipefail
STACK_DIR=${1:?thư mục chứa docker-compose.yml}
DEST=${2:?thư mục đích}
KEEP_DAYS=${KEEP_DAYS:-30}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$DEST/$STAMP"
mkdir -p "$OUT"
cd "$STACK_DIR"
set -a; [ -f .env ] && . ./.env; set +a

docker compose exec -T agrimind tar -C /var/lib/agrimind -czf - corpus audit > "$OUT/agrimind-data.tgz"
cp clients.yaml "$OUT/clients.yaml"

# Bản chụp Qdrant qua API nội bộ (gọi từ trong container agrimind vì Qdrant không mở cổng ra ngoài)
docker compose exec -T -e QK="$QDRANT_API_KEY" agrimind python - <<'PY' > "$OUT/qdrant-snapshot.txt"
import os, httpx
h = {"api-key": os.environ["QK"]}
r = httpx.post("http://qdrant:6333/collections/agrimind_corpus/snapshots", headers=h, timeout=600)
print(r.status_code, r.text)
PY

sha256sum "$OUT"/* > "$OUT/SHA256SUMS"
find "$DEST" -mindepth 1 -maxdepth 1 -type d -mtime +"$KEEP_DAYS" -exec rm -rf {} +
echo "Đã sao lưu vào $OUT"
