##############################################################################
# Stage 1 — Build deps (cached layer)
##############################################################################
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

##############################################################################
# Stage 2 — Runtime image
##############################################################################
FROM python:3.11-slim AS runtime

LABEL maintainer="Fraud Detection Engine <fraud-eng@company.com>"
LABEL version="1.0.0"
LABEL description="Real-Time Fraud Detection API"

WORKDIR /app

# Runtime system deps (LightGBM / XGBoost shared libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/      ./src/
COPY data/     ./data/
COPY scripts/  ./scripts/
COPY .env.example .env

# Create runtime directories
RUN mkdir -p models_store logs/drift_reports

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info", \
     "--access-log"]
