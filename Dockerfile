FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY agent_engine ./agent_engine
RUN pip install --no-cache-dir ".[openai,anthropic]"
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "agent_engine.server:app", "--host", "0.0.0.0", "--port", "8000"]
