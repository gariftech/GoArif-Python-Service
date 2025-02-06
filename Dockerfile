# Stage 1: Install dependencies
FROM python:3.9-slim AS deps
WORKDIR /app

# Install system dependencies and clone repository in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    apt-transport-https \
    ca-certificates \
    gnupg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification /indonesia-bert-sentiment-classification \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update -y \
    && apt-get install google-cloud-cli -y \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir transformers huggingface-hub urllib3

# Stage 2: Build application
FROM deps AS builder
WORKDIR /app
COPY . .
RUN mkdir -p /app/static

# Stage 3: Final stage
FROM python:3.9-slim AS runner
WORKDIR /app

RUN pip install fastapi uvicorn

# Copy the gcloud configuration from deps to runner stage
COPY --from=deps /root/.config/gcloud /root/.config/gcloud

# Copy only necessary files from previous stages
COPY --from=builder /app /app
COPY --from=builder /indonesia-bert-sentiment-classification /indonesia-bert-sentiment-classification

# Copy the necessary Python packages from deps to runner stage
COPY --from=deps /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

EXPOSE 9000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
