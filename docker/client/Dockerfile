# Dockerfile.client
FROM python:3.10.12-slim

WORKDIR /app

# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY t4fids/client /app/client
COPY t4fids/common /app/common
COPY scripts/run_client.py /app/run_client.py

ENTRYPOINT ["python", "run_client.py"]