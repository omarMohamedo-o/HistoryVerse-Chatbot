FROM docker:27-cli-alpine

RUN apk add --no-cache bash coreutils curl util-linux findutils

COPY disk-guardian.sh /usr/local/bin/disk-guardian.sh
RUN chmod +x /usr/local/bin/disk-guardian.sh

RUN addgroup -S guardian && adduser -S -G guardian guardian

USER guardian

ENTRYPOINT ["/usr/local/bin/disk-guardian.sh"]
