# Dockerfile.client
FROM python:3.10.12-slim

WORKDIR /app

# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY t4fids/server /app/server
COPY t4fids/common /app/common
COPY scripts/run_server.py /app/run_server.py

EXPOSE 8080

ENTRYPOINT ["python", "run_server.py"]