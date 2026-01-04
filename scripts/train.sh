#!/bin/bash
# CoCoBind Training Script
# Usage: bash scripts/train.sh

set -e

# Configuration
SPLITS=("unseen_pair" "unseen_rna" "unseen_compound" "unseen_both")
FOLDS=(0 1 2 3 4)
GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-"./data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs"}

echo "======================================"
echo "CoCoBind Training"
echo "======================================"
echo "GPU: $GPU"
echo "DATA_ROOT: $DATA_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# Train all splits and folds
for split in "${SPLITS[@]}"; do
    echo "Training split: $split"
    for fold in "${FOLDS[@]}"; do
        echo "  Fold: $fold"
        python -m cocobind.train \
            --split "$split" \
            --fold "$fold" \
            --data_root "$DATA_ROOT" \
            --output_dir "$OUTPUT_DIR/base/$split/fold$fold" \
            --gpu "$GPU"
    done
done

echo ""
echo "Training completed for all splits and folds!"
