FROM python:3.10

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libfreetype6-dev \
    libpng-dev \
    libzmq3-dev \
    pkg-config \
    python3-dev \
    unzip \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m textblob.download_corpora

EXPOSE 8501

CMD ["streamlit", "run", "eur_usd_predictor.py", "--server.port=10000", "--server.enableCORS=false"]
