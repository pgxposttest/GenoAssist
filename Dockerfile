# Use an appropriate base image
FROM python:3.10-slim

# Run Python unbuffered for clean logs
ENV PYTHONUNBUFFERED 1

# Install system build dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential gcc g++ python3-dev

# Upgrade pip & setuptools
RUN pip install -U pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements & install
COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

# Copy app code
COPY . /app/

EXPOSE 8080

# Start the Chainlit app
CMD ["python", "-m", "chainlit", "run", "model.py", "-h", "--port", "8080"]