FROM python:3.12-slim
WORKDIR /app
RUN useradd --create-home --uid 10001 agrimind
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir '.[pdf]'
COPY data/kg ./data/kg
COPY data/twin ./data/twin
COPY tests/fixtures ./tests/fixtures
# Mặc định chạy được thử với tài liệu mẫu; chạy thật thì gắn kho thật vào /data/corpus.
ENV AGRIMIND_ROLE=country AGRIMIND_COUNTRY=VN AGRIMIND_KG_DIR=/app/data/kg AGRIMIND_TWIN_RULES=/app/data/twin PYTHONUNBUFFERED=1
USER agrimind
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "agrimind.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
