#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <input_sparse_model> <output_text_model>" >&2
  exit 2
fi

INPUT=$1
OUTPUT=$2
mkdir -p "$OUTPUT"

if ! command -v colmap >/dev/null 2>&1; then
  echo "COLMAP executable not found." >&2
  exit 1
fi

colmap model_converter \
  --input_path "$INPUT" \
  --output_path "$OUTPUT" \
  --output_type TXT

echo "Exported COLMAP text model to $OUTPUT"
echo "Note: standard COLMAP export does not preserve GlueMap virtual-track provenance."
