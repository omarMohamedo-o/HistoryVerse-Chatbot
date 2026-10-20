# Disk Guard and Docker Cleanup Architecture

This document describes the automated storage maintenance and disk protection daemon inspired by the `pipeline-infra` guardian architecture.

## Overview

HistoryVerse processes multimodal files (PDFs, images, audio files, and model weights) that can rapidly consume host and container disk space.
To guarantee continuous operations and prevent host failures due to out-of-disk (OOD) scenarios, HistoryVerse incorporates:

1. **`disk-guardian` Service** (`services/disk-guardian/`):
   - Continuous background daemon monitoring disk capacity and inode usage.
   - Triggers automated pruning when disk exceeds `DISK_WARN_PCT` (default: 85%) or `DISK_CRIT_PCT` (default: 90%).
   - Safely removes dangling images, unused builder caches, and stopped containers older than 24 hours.
   - Cleans transient audio recordings (`outputs/*.mp3`, `outputs/*.wav`) older than 24 hours.
   - Dispatches emergency telemetry alerts to Prometheus Alertmanager.

2. **`docker-clean.sh` Script** (`scripts/docker-clean.sh`):
   - Standalone CLI utility for manual operations or cron schedules.
   - Supports `--dry-run` and configurable age filters (`PRUNE_UNTIL_HOURS`).

## Running Disk Guardian

Included in `docker-compose.monitoring.yml`:

```bash
docker compose -f docker-compose.monitoring.yml up -d disk-guardian
```

## Manual Cleanup Command

```bash
./scripts/docker-clean.sh
```
