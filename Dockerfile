FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN python -m pip install --upgrade pip && python -m pip install .

RUN mkdir -p /app/artifacts && chown -R app:app /app

USER app

RUN python scripts/train.py --users 900 --hotels 260 --interactions 30000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "hotel_recommender.api:app", "--host", "0.0.0.0", "--port", "8000"]
