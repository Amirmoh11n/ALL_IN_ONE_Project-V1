#Dockerfile for hugging face
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/artifacts/exports/brain_tumor_efficientnet_b3.onnx \
    MAX_UPLOAD_MB=10 \
    CORS_ORIGINS=*


COPY requirements-web.txt .
RUN pip install --upgrade pip && pip install -r requirements-web.txt

COPY configs ./configs
COPY webapplication ./webapplication
COPY src ./src
COPY artifacts/exports/brain_tumor_efficientnet_b3.onnx ./artifacts/exports/


EXPOSE 7860
CMD ["uvicorn", "webapplication.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]