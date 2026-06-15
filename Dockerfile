FROM node:20-alpine AS trace-frontend-build

WORKDIR /frontend

COPY trace_dashboard/package.json trace_dashboard/package-lock.json ./
RUN npm ci

COPY trace_dashboard/ ./
RUN npm run build

FROM python:3.11-slim AS python-runtime

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# uv + external fetch MCP server (consumed as an MCP client over stdio)
RUN pip install --no-cache-dir uv && uv tool install mcp-server-fetch

COPY agent/ ./agent/
COPY app/ ./app/
COPY services/ ./services/
COPY eval/ ./eval/

CMD ["python", "-m", "app.bot"]

FROM python-runtime AS trace-runtime

COPY --from=trace-frontend-build /frontend/dist ./trace_dashboard/dist

CMD ["uvicorn", "app.trace_api:app", "--host", "0.0.0.0", "--port", "8010"]
