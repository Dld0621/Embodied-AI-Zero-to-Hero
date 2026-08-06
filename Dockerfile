FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements-core.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-core.txt

COPY . .

CMD ["python", "scripts/run_pipeline.py", "--list"]
