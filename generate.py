"""
generate.py — Python alternative to generate.mjs

Usage:
    pip install sentence-transformers
    python generate.py

Output:
    public/embeddings.bin  (~6.3 MB, Float32, shape [4310 × 384])
"""

import json, os, time, struct
import numpy as np
from sentence_transformers import SentenceTransformer

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load data
with open(os.path.join(script_dir, "public", "data.json"), encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} idioms")
texts = [d["e"] for d in data]   # pre-built embed texts

# Embed
print("Loading model…")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print(f"Embedding {len(texts)} idioms…")
t0 = time.time()
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
).astype(np.float32)

print(f"Done in {time.time()-t0:.1f}s  |  shape: {embeddings.shape}")

# Write binary
out = os.path.join(script_dir, "public", "embeddings.bin")
embeddings.tofile(out)
mb = os.path.getsize(out) / 1024 / 1024
print(f"\n✅  public/embeddings.bin  ({mb:.1f} MB)")
print("   Deploy the public/ folder to Cloudflare Pages.")
