# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the Hugging Face repository into a separate directory
RUN git clone https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification /indonesia-bert-sentiment-classification

# Install Hugging Face Transformers and other dependencies
RUN pip install --upgrade pip
RUN pip install transformers huggingface-hub

# Copy the current directory contents into the container at /app
COPY . /app

# Dockerfile
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN echo "Using token: ${HF_TOKEN}

# Ensure the static folder exists
RUN mkdir -p /app/static

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir urllib3 \
    && pip install --no-cache-dir -r requirements.txt

# Expose port 9000
EXPOSE 9000

# Run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]

