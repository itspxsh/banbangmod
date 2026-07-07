#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <image-ref>" >&2
  exit 1
fi

docker tag gemma4-track2-agent:latest "$1"
docker push "$1"
