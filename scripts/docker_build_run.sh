#!/usr/bin/env bash
set -euo pipefail
docker build -t credit-scoring-api .
docker run --rm -p 8000:8000 credit-scoring-api
