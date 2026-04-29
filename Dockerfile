FROM python:3.12-slim

# Step 1: Set working directory
WORKDIR /code

# Step 2: Prevent Python from buffering logs (important for Docker)
ENV PYTHONUNBUFFERED=1

# Step 3: Install system-level dependencies
RUN apt-get update && apt-get install -y \
    # Postgres
    libpq-dev \
    gcc \
    # Tesseract OCR engine + English language pack
    tesseract-ocr \
    tesseract-ocr-eng \
    # Ghostscript for camelot table extraction
    ghostscript \
    # OpenCV headless runtime deps
    libgl1 \
    libglib2.0-0 \
    # Camelot needs these for PDF rendering
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Install Python libraries (Cached Layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm
# Step 5: Copy your project code
COPY . .

# Step 6: Start the server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}", "--reload"]