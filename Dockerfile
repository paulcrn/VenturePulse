FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

CMD ["sh", "-c", "uvicorn fast_api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
