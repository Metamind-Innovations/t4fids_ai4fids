# Dockerfile.client
FROM python:3.10.12-slim

WORKDIR /app

# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY t4fids/client /app/t4fids/client
COPY t4fids/common /app/t4fids/common
COPY scripts/run_client.py /app/run_client.py

CMD ["python", "run_client.py", "--config", "config.json"]