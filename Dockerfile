ARG WEAVEDESK_DEPS_IMAGE=python:3.11-slim
FROM ${WEAVEDESK_DEPS_IMAGE} AS dependencies
USER root
WORKDIR /build/weavedesk
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 WEAVEDESK_DATA=/var/lib/weavedesk
WORKDIR /opt/weavedesk
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY weavedesk ./weavedesk
COPY static ./static
RUN useradd -U -u 11513 weavedesk && install -d -o weavedesk -g weavedesk /var/lib/weavedesk
USER weavedesk
EXPOSE 8000
CMD ["python", "-m", "weavedesk"]
