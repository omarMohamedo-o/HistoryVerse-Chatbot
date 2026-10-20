FROM python:3.11-alpine

RUN apk add --no-cache curl ca-certificates

WORKDIR /app
COPY services/heartbeat/heartbeat.py /app/heartbeat.py

RUN addgroup -S guardian && adduser -S -G guardian guardian
USER guardian

ENTRYPOINT ["python", "/app/heartbeat.py"]
