# Mimir News Scraper - Multi-stage Production Dockerfile
# Base: Python 3.11 slim for optimal size and security

ARG PYTHON_VERSION=3.11.7
ARG ALPINE_VERSION=3.19

# ================================
# Stage 1: Build Dependencies
# ================================
FROM python:${PYTHON_VERSION}-alpine${ALPINE_VERSION} AS builder

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    postgresql-dev \
    libffi-dev \
    openssl-dev \
    cargo \
    rust \
    g++ \
    make \
    linux-headers

# Create app user and directory
RUN addgroup -g 1001 -S mimir && \
    adduser -S -D -u 1001 -G mimir mimir

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt requirements_api.txt ./
RUN pip install --no-cache-dir --user --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements.txt && \
    pip install --no-cache-dir --user -r requirements_api.txt

# ================================
# Stage 2: NLP Models Download
# ================================
FROM python:${PYTHON_VERSION}-alpine${ALPINE_VERSION} AS nlp-models

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Install minimal system dependencies for spaCy
RUN apk add --no-cache gcc musl-dev

# Download spaCy language models
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download nl_core_news_sm && \
    python -m spacy download de_core_news_sm && \
    python -m spacy download fr_core_news_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', download_dir='/nltk_data'); nltk.download('stopwords', download_dir='/nltk_data')"

# ================================
# Stage 3: Production Runtime
# ================================
FROM python:${PYTHON_VERSION}-alpine${ALPINE_VERSION} AS production

# Security: Create non-root user
RUN addgroup -g 1001 -S mimir && \
    adduser -S -D -u 1001 -G mimir mimir

# Install only runtime dependencies
RUN apk add --no-cache \
    postgresql-client \
    libpq \
    libffi \
    openssl \
    ca-certificates \
    curl \
    # For healthchecks
    wget \
    # For potential debugging (can be removed in production)
    dumb-init

# Copy Python packages from builder
COPY --from=builder /root/.local /home/mimir/.local
COPY --from=nlp-models /root/.local/lib/python3.11/site-packages/en_core_web_sm /home/mimir/.local/lib/python3.11/site-packages/en_core_web_sm
COPY --from=nlp-models /root/.local/lib/python3.11/site-packages/nl_core_news_sm /home/mimir/.local/lib/python3.11/site-packages/nl_core_news_sm
COPY --from=nlp-models /root/.local/lib/python3.11/site-packages/de_core_news_sm /home/mimir/.local/lib/python3.11/site-packages/de_core_news_sm
COPY --from=nlp-models /root/.local/lib/python3.11/site-packages/fr_core_news_sm /home/mimir/.local/lib/python3.11/site-packages/fr_core_news_sm
COPY --from=nlp-models /nltk_data /home/mimir/nltk_data

# Set up environment
ENV PATH=/home/mimir/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NLTK_DATA=/home/mimir/nltk_data \
    # Production settings
    ENVIRONMENT=production \
    # Security
    PYTHONHASHSEED=random

# Create application directories
WORKDIR /app
RUN mkdir -p /app/logs /app/data /app/static /app/templates && \
    chown -R mimir:mimir /app

# Copy application code
COPY --chown=mimir:mimir . .

# Remove unnecessary files for production
RUN rm -rf \
    tests/ \
    docs/ \
    *.md \
    .git* \
    __pycache__/ \
    *.pyc \
    *.pyo \
    *.pyd \
    .Python \
    .pytest_cache/ \
    .coverage

# Set proper permissions
RUN chmod +x /app/*.py && \
    find /app -type f -name "*.py" -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \;

# Switch to non-root user
USER mimir

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose ports
EXPOSE 5000 5001

# Use dumb-init for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Default command (can be overridden)
CMD ["python", "web_interface.py"]