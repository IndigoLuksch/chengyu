#!/bin/bash
# Upload all model files to R2 bucket so they're served alongside model_quantized.onnx
# Run from project root: bash upload-model-to-r2.sh

BUCKET="chengyu-models"  # adjust to your R2 bucket name
MODEL_DIR="public/models/Xenova/bge-small-en-v1.5"

for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json; do
  echo "Uploading $f ..."
  wrangler r2 object put "$BUCKET/$f" --file "$MODEL_DIR/$f" --content-type "application/json"
done

echo "Done. Files available at: https://pub-28d1ded41c3d4f0bb2ee8f4ff66d908b.r2.dev/<filename>"
