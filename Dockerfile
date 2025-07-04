FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential gcc g++ python3-dev

RUN pip install -U pip setuptools wheel

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/

COPY vectorstore /app/vectorstore

EXPOSE 8080

CMD ["chainlit", "run", "model.py", "--host", "0.0.0.0", "--port", "8080"]