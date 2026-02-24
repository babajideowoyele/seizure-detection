# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --default-timeout=300 --retries=5 torch==2.2.2
RUN pip install --no-cache-dir --default-timeout=300 --retries=5 -r requirements.txt

VOLUME ["/data"]
VOLUME ["/output"]

ENV INPUT=""
ENV OUTPUT="test_data.csv"

CMD python3 main.py "/data/$INPUT" "/output/$OUTPUT"
