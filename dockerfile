# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git

# Clone the Hugging Face repository into a separate directory
RUN git clone https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification /indonesia-bert-sentiment-classification

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install urllib3 --upgrade
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 9000
EXPOSE 9000

# Run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]