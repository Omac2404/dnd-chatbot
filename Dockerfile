FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# requirements'ı kopyala
COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt


# Tüm projeyi kopyala (alt klasör dahil)
COPY . .

# ÇALIŞMA DİZİNİNİ SENİN GERÇEK KODUNA GÖRE AYARLA
WORKDIR /app/src


EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
