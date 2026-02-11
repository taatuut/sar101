#!/usr/bin/env bash

# Exit on error
set -e

LOG_DIR="./logs"
OUT_DIR="./outputs"

echo "Cleaning log and output files..."

# Remove *.log from ./logs
if [ -d "$LOG_DIR" ]; then
  find "$LOG_DIR" -type f -name "*.log" -print -delete
else
  echo "Directory $LOG_DIR does not exist."
fi

# Remove *.tif, *.png, *.gpkg, *.xml from ./outputs
if [ -d "$OUT_DIR" ]; then
  find "$OUT_DIR" -type f \( -name "*.tif" -o -name "*.png" -o -name "*.gpkg"  -o -name "*.xml" \) -print -delete
else
  echo "Directory $OUT_DIR does not exist."
fi

echo "Cleanup complete."
