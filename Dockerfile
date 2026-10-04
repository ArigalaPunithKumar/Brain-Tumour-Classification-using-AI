FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

CMD gunicorn --workers 1 --threads 2 --timeout 180 --bind 0.0.0.0:$PORT app:app
