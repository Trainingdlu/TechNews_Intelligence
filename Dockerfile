FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/
COPY app/ ./app/
COPY services/ ./services/
COPY eval/ ./eval/

CMD ["python", "app/bot.py"]
