# Build once locally; no runtime downloads. Only model-authored code may run here.
FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
    sympy==1.14.0 z3-solver==4.16.0.0
USER 65534:65534
WORKDIR /work
