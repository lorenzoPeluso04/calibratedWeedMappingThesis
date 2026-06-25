#!/usr/bin/env bash
# run_benchmark.sh — esegue il benchmark superpixel (20 e 200 segmenti)
# Uso: bash run_benchmark.sh [--image-dir DIR] [--ground-truth-dir DIR]
#
# Valori di default: le directory del dataset RoWeeder già usate nel progetto.

set -euo pipefail

IMAGE_DIR="${IMAGE_DIR:-RoWeeder/dataset/patches/512/003/RGB}"
GT_DIR="${GT_DIR:-RoWeeder/dataset/patches/512/003/groundtruth}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results}"
GSD="${GSD:-0.05}"
ACCURACY="${ACCURACY:-0.8}"

# Permette di sovrascrivere le directory via argomenti posizionali
while [[ $# -gt 0 ]]; do
    case "$1" in
        --image-dir)       IMAGE_DIR="$2";  shift 2 ;;
        --ground-truth-dir) GT_DIR="$2";   shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --gsd-m)           GSD="$2";        shift 2 ;;
        --accuracy)        ACCURACY="$2";   shift 2 ;;
        *) echo "Opzione sconosciuta: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Benchmark Superpixel — avvio"
echo "  Image dir  : $IMAGE_DIR"
echo "  GT dir     : $GT_DIR"
echo "  Output dir : $OUTPUT_DIR"
echo "  GSD        : $GSD m/px"
echo "  Accuracy   : $ACCURACY"
echo "============================================================"

# ---------- 20 superpixel ----------
echo ""
echo ">>> [1/2] Benchmark con 20 superpixel..."
python main.py benchmark-superpixels \
    --image-dir        "$IMAGE_DIR" \
    --ground-truth-dir "$GT_DIR" \
    --output-dir       "$OUTPUT_DIR" \
    --num-segments     20 \
    --gsd-m            "$GSD" \
    --accuracy         "$ACCURACY"

echo ""
echo ">>> [1/2] Benchmark con 20 superpixel completato."

# ---------- 200 superpixel ----------
echo ""
echo ">>> [2/2] Benchmark con 200 superpixel..."
python main.py benchmark-superpixels \
    --image-dir        "$IMAGE_DIR" \
    --ground-truth-dir "$GT_DIR" \
    --output-dir       "$OUTPUT_DIR" \
    --num-segments     200 \
    --gsd-m            "$GSD" \
    --accuracy         "$ACCURACY"

echo ""
echo ">>> [2/2] Benchmark con 200 superpixel completato."

echo ""
echo "============================================================"
echo "  Benchmark completato. Risultati in: $OUTPUT_DIR"
echo "============================================================"
