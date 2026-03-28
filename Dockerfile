FROM python:3.11-slim

# Instalar ffmpeg e dependências de sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar setuptools+wheel ANTES para que openai-whisper compile corretamente
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-build-isolation -r requirements.txt

# Copiar código
COPY . .

# Criar diretórios de dados
RUN mkdir -p /data/uploads /data/processed /data/jingles

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
