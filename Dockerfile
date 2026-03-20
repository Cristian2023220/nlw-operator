FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY . /app/
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "[IP_ADDRESS]", "--port", "8000"]