FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements-index.txt requirements-rag.txt ./
RUN pip install --no-cache-dir -r requirements-rag.txt

COPY scripts/ scripts/
COPY knowledge_base/ knowledge_base/
COPY vector_index/ vector_index/

CMD ["python", "scripts/rag_bot.py"]
