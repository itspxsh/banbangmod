#!/usr/bin/env bash
set -euo pipefail

docker buildx build --platform linux/amd64 -t gemma4-track2-agent:latest .
