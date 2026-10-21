#!/bin/sh
set -eu

if [ -d /state ]; then
  chown -R guardian:guardian /state 2>/dev/null || true
fi

exec su-exec guardian "$@"
