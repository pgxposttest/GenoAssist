# Use an appropriate base image, e.g., python:3.10-slim
FROM python:3

# Set environment variables (e.g., set Python to run in unbuffered mode)
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install g++ -y

# 👇 STYLE YOUR DOCKERFILE LIKE A PRO
RUN pip install -U \
    pip \
    setuptools \
    wheel

# Set the working directory
WORKDIR /app

# Copy your application's requirements and install them
COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

# Copy your application code into the container
COPY . /app/

EXPOSE 8080

CMD ["python", "-m", "chainlit", "run", "model.py", "-h", "--port", "8080"]