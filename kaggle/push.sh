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
  all)
    cd all && kaggle kernels push
    ;;
  status-all)
    kaggle kernels status vaibhavnarla/tinystories-all
    ;;
  pull-all)
    kaggle kernels output vaibhavnarla/tinystories-all -p ./out-all
    ;;
  status)
    kaggle kernels list --mine --page-size 15
    ;;
  pull-prep)
    kaggle kernels output vaibhavnarla/tinystories-prep-data -p ./out-prep
    ;;
  pull-train)
    kaggle kernels output vaibhavnarla/tinystories-train-25m -p ./out-train
    ;;
  *)
    echo "usage: $0 {prep|train|all|status|status-all|pull-prep|pull-train|pull-all}"
    exit 1
    ;;
esac
