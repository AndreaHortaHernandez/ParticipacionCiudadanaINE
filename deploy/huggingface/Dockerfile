# Space de Hugging Face (SDK: Docker) para la app de riesgo de abstención.
# No hace falta subir datos al Space: el código se clona de GitHub y los
# resultados (modelo + mapas procesados) se bajan del Release al construir.
# Para actualizar la app: en el Space, Settings → Factory rebuild.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face corre el contenedor con el usuario 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user

ARG REPO=AndreaHortaHernandez/ParticipacionCiudadanaINE
# Estos ADD cambian cuando hay commits o un Release nuevo: invalidan la caché de Docker
ADD --chown=user https://api.github.com/repos/${REPO}/git/refs/heads/main /tmp/commit.json
RUN git clone --depth 1 https://github.com/${REPO}.git app

WORKDIR /home/user/app
RUN pip install --no-cache-dir --user -r deploy/huggingface/requirements-app.txt

ADD --chown=user https://api.github.com/repos/${REPO}/releases/latest /tmp/release.json
RUN curl -fsSL -o /tmp/r.zip https://github.com/${REPO}/releases/latest/download/resultados_para_colab.zip \
    && unzip -q /tmp/r.zip -d /tmp/r \
    && mkdir -p data/processed \
    && cp -r "$(dirname "$(find /tmp/r -name 'riesgo_2027.parquet' | head -1)")"/. data/processed/ \
    && rm -rf /tmp/r /tmp/r.zip \
    && ls data/processed data/processed/geo | head -40

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
