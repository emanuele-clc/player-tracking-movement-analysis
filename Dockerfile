# Hugging Face Spaces now deploys Streamlit apps via the Docker SDK (the
# built-in "Streamlit" SDK option was deprecated - see
# https://huggingface.co/docs/hub/spaces-sdks-streamlit). This Dockerfile is
# the standard HF-recommended shape: non-root user 1000, deps installed
# before copying the rest of the app so Docker can cache that layer, app
# served on port 8501 (matches app_port in README.md's YAML block).
#
# Also runs fine as a plain local Docker image if you'd rather not use
# Streamlit's own dev server directly:
#   docker build -t player-tracking .
#   docker run -p 8501:8501 player-tracking

FROM python:3.11-slim

# opencv (even the headless build) needs these two shared libs at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

RUN pip install --no-cache-dir --upgrade pip

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
