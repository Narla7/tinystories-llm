#!/usr/bin/env bash
# Push / manage the Kaggle kernels. Requires ~/.kaggle/kaggle.json
#   kaggle.com -> your avatar -> Settings -> Create New API Token
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in
  prep)
    cd prep && kaggle kernels push
    ;;
  train)
    cd train && kaggle kernels push
    ;;
  status)
    kaggle kernels list --mine --page-size 15
    ;;
  pull-prep)
    kaggle kernels output narla/tinystories-prep-data -p ./out-prep
    ;;
  pull-train)
    kaggle kernels output narla/tinystories-train-25m -p ./out-train
    ;;
  *)
    echo "usage: $0 {prep|train|status|pull-prep|pull-train}"
    exit 1
    ;;
esac
