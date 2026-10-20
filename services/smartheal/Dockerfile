FROM python:3.11-alpine

RUN apk add --no-cache bash curl docker-cli su-exec

WORKDIR /app
COPY services/smartheal/smartheal.py /app/smartheal.py

RUN addgroup -S guardian && adduser -S -G guardian guardian

COPY services/smartheal/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh", "python", "/app/smartheal.py"]
