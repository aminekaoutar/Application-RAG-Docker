FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional data processing dependencies
RUN pip install --no-cache-dir \
    beautifulsoup4 \
    requests \
    faiss-cpu \
    langchain-huggingface

COPY . .

# Default command - can be overridden
CMD ["python", "-c", "print('Data processing container ready')"]