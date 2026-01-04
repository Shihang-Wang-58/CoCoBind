# CoCoBind Training Script (PowerShell)
# Usage: .\scripts\train.ps1

param(
    [string]$GPU = "0",
    [string]$DataRoot = ".\data",
    [string]$OutputDir = ".\outputs",
    [string]$Split = "",
    [int]$Fold = -1
)

$ErrorActionPreference = "Stop"

$SPLITS = @("unseen_pair", "unseen_rna", "unseen_compound", "unseen_both")
$FOLDS = @(0, 1, 2, 3, 4)

Write-Host "======================================"
Write-Host "CoCoBind Training"
Write-Host "======================================"
Write-Host "GPU: $GPU"
Write-Host "DATA_ROOT: $DataRoot"
Write-Host "OUTPUT_DIR: $OutputDir"
Write-Host ""

# If specific split/fold provided, train only that
if ($Split -ne "" -and $Fold -ge 0) {
    Write-Host "Training split: $Split, fold: $Fold"
    python -m cocobind.train `
        --split $Split `
        --fold $Fold `
        --data_root $DataRoot `
        --output_dir "$OutputDir\base\$Split\fold$Fold" `
        --gpu $GPU
}
else {
    # Train all splits and folds
    foreach ($split in $SPLITS) {
        Write-Host "Training split: $split"
        foreach ($fold in $FOLDS) {
            Write-Host "  Fold: $fold"
            python -m cocobind.train `
                --split $split `
                --fold $fold `
                --data_root $DataRoot `
                --output_dir "$OutputDir\base\$split\fold$fold" `
                --gpu $GPU
        }
    }
}

Write-Host ""
Write-Host "Training completed!"
