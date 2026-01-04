#!/bin/bash
# CoCoBind Evaluation Script
# Usage: bash scripts/eval.sh

set -e

# Configuration
SPLITS=("unseen_pair" "unseen_rna" "unseen_compound" "unseen_both")
FOLDS=(0 1 2 3 4)
GPU=${GPU:-0}
DATA_ROOT=${DATA_ROOT:-"./data"}
CKPT_DIR=${CKPT_DIR:-"./outputs/base"}

echo "======================================"
echo "CoCoBind Evaluation"
echo "======================================"
echo "GPU: $GPU"
echo "CKPT_DIR: $CKPT_DIR"
echo ""

# Evaluate all splits and folds
for split in "${SPLITS[@]}"; do
    echo "Evaluating split: $split"
    for fold in "${FOLDS[@]}"; do
        CHECKPOINT="$CKPT_DIR/$split/fold$fold/best.pt"
        if [ -f "$CHECKPOINT" ]; then
            echo "  Fold: $fold"
            python -m cocobind.eval \
                --checkpoint "$CHECKPOINT" \
                --split "$split" \
                --fold "$fold" \
                --data_root "$DATA_ROOT" \
                --gpu "$GPU" \
                --save_preds
        else
            echo "  Fold: $fold - checkpoint not found, skipping"
        fi
    done
done

echo ""
echo "Evaluation completed!"
