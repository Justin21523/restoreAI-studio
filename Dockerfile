FROM node:22-alpine AS web
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY web ./web
COPY vite.config.ts ./
RUN VITE_APP_MODE=real npm run build

FROM python:3.10-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY restorai ./restorai
RUN pip install --no-cache-dir .
COPY alembic.ini ./
COPY alembic ./alembic
COPY --from=web /app/dist ./dist

EXPOSE 8000
CMD ["uvicorn", "restorai.api:app", "--host", "0.0.0.0", "--port", "8000"]
