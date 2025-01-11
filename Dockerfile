# Stage 1: Install dependencies
FROM python:3.9-slim AS builder
WORKDIR /app

# Install system dependencies and clone repository in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification /indonesia-bert-sentiment-classification \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir transformers huggingface-hub urllib3

# Stage 2: Build application
FROM builder AS application
WORKDIR /app
COPY . .
RUN mkdir -p /app/static

# Stage 3: Final stage
FROM python:3.9-slim
WORKDIR /app

RUN pip install fastapi uvicorn

# Copy only necessary files from previous stages
COPY --from=application /app /app
COPY --from=application /indonesia-bert-sentiment-classification /indonesia-bert-sentiment-classification
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

EXPOSE 9000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
