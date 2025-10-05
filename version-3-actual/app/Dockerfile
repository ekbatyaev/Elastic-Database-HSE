FROM python:3.10-slim

# Устанавливаем необходимые системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    python3-dev \
    curl \
    libgl1 \
    libglib2.0-0 \
    # Для pymupdf (fitz)
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем модель spaCy (более надежный метод)
RUN python -m spacy download ru_core_news_sm && \
    python -m spacy validate

# Копируем весь код за один раз
COPY . .

# Запуск FastAPI-приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]