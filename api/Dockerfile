FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p /app/api/models

# Copy code
# Assuming context is project root
COPY api/ /app/api/

ENV PYTHONPATH=/app
ENV PORT=8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
