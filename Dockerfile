FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml ./
RUN pip install uv && uv sync

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.transaction_splitting.api.main:app", "--host", "0.0.0.0", "--port", "8000"]