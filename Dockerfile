FROM python:3.11-slim

WORKDIR /app/agents

COPY agents/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents/ .

CMD ["python", "bot.py"]