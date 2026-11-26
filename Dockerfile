FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (models and X_test.csv are generated during build so fresh clones work)
COPY src/ src/
COPY data/ames_housing.csv data/ames_housing.csv
COPY reports/ reports/
COPY app.py .
COPY pyproject.toml .

# Train models inside image so models/*.joblib and data/X_test.csv exist at runtime
RUN python -m src.train && rm -rf mlruns

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check using Python stdlib (curl is not installed in python:3.11-slim)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
