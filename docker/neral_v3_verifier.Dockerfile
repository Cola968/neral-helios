# Dedicated synthetic verification container. Network is disabled at RUN time.
# Linux image build may use network to fetch pinned public dependencies.
FROM python:3.12.8-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/source
RUN python -m pip install --disable-pip-version-check --no-cache-dir \
    sympy==1.14.0 networkx==3.6.1 z3-solver==4.16.0.0
WORKDIR /source
USER 65534:65534
