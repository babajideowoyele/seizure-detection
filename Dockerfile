# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN pip install --no-cache-dir -r requirements.txt

USER appuser

VOLUME ["/data"]
VOLUME ["/output"]

ENV INPUT=""
ENV OUTPUT="test_data.csv"

CMD python3 main.py "/data/$INPUT" "/output/$OUTPUT"
