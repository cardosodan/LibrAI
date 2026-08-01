# Imagem pra Hugging Face Spaces (SDK Docker) — mesma stack Python 3.14 usada
# em dev local, empacotada pra rodar como servidor de verdade.
FROM python:3.14-slim

# opencv-contrib-python precisa dessas libs de sistema num Linux mínimo —
# sem elas, o import falha com "libGL.so.1: cannot open shared object file"
# mesmo com o pacote pip instalado certinho.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torch precisa do índice CPU dedicado (senão baixa a build CUDA, bem maior
# e desnecessária aqui) — mesmo passo documentado no README pra dev local.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Baixa os bundles do MediaPipe (~20MB, gitignored) durante o BUILD da
# imagem — ficam prontos dentro da imagem, sem precisar baixar de novo a
# cada vez que o container reinicia.
RUN PYTHONPATH=src python -m sinalizai.baixar_modelos_mediapipe

# Hugging Face Spaces (SDK Docker) espera o servidor respondendo na 7860.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --timeout 120 app:app"]
