FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV TRADING_REGION=us
WORKDIR /app

COPY requirements.production.txt .
RUN pip install --no-cache-dir -r requirements.production.txt

COPY . .

# Default to API server; Railway services can override the command.
CMD ["uvicorn", "scripts.production.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
