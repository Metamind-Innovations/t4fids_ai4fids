# Dockerfile.client
FROM python:3.10.12-slim

WORKDIR /app

# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY t4fids/server /app/t4fids/server
COPY t4fids/common /app/t4fids/common
COPY scripts/run_server.py /app/run_server.py

EXPOSE 8080

CMD ["python", "run_server.py", "--config", "config.json"]
