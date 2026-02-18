FROM python:3.10-slim

ARG INSTALL_TORCH=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-torch.txt /app/
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt
RUN if [ "$INSTALL_TORCH" = "1" ]; then python -m pip install -r requirements-torch.txt; fi

COPY . /app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "news_sentinel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
