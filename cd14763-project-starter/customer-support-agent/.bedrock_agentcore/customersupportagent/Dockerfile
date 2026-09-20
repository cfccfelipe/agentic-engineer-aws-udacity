FROM --platform=linux/arm64 python:3.12-slim-bookworm

WORKDIR /app

# Instalar uv desde la imagen oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Variables de entorno globales del contenedor
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1 \
    AWS_REGION=us-east-1 \
    AWS_DEFAULT_REGION=us-east-1 \
    AGENTCORE_PROFILE_ID=browser_profile_1x6ad-1lTDsfeNFy \
    AGENTCORE_SUPPRESS_RECOMMENDATION=1

# 1. Copiar primero los archivos de dependencias para aprovechar la caché
COPY pyproject.toml uv.lock ./

# 2. Instalar dependencias
RUN uv pip install . && \
    uv pip install aws-opentelemetry-distro==0.12.2

# 3. Copiar código fuente
COPY . .

# 4. Usuario no raíz
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 8080
EXPOSE 8000
EXPOSE 9000

CMD ["opentelemetry-instrument", "python", "-m", "main"]
