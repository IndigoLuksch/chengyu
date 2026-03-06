"""
filter_chinese.py — remove non-Chinese entries from data.json + embeddings.bin
without re-running the slow embedding step.
Usage: python filter_chinese.py
"""
from datasets import load_dataset
import json, os, struct, numpy as np

BASE = os.path.dirname(__file__)

# ── Get valid Chinese idioms from dataset ─────────────────────────────────────
print("Loading dataset to get Chinese idiom list…")
ds = load_dataset("kenantang/IdiomTranslate30", split="train")
en = ds.filter(lambda x: x["source_language"] == "Chinese" and x["target_language"] == "English")
chinese_set = set(en["idiom"])
print(f"  {len(chinese_set):,} valid Chinese idioms")

# ── Load current data.json ────────────────────────────────────────────────────
data_path = os.path.join(BASE, "public", "data.json")
data = json.load(open(data_path))
print(f"  {len(data):,} entries in data.json")

# ── Find indices to keep ──────────────────────────────────────────────────────
keep = [i for i, d in enumerate(data) if d["c"] in chinese_set]
print(f"  Keeping {len(keep):,}, removing {len(data)-len(keep):,}")

# ── Filter data.json ──────────────────────────────────────────────────────────
filtered = [data[i] for i in keep]
with open(data_path, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, separators=(",", ":"))
mb = os.path.getsize(data_path) / 1024 / 1024
print(f"✅  data.json → {mb:.1f} MB")

# ── Filter embeddings.bin ─────────────────────────────────────────────────────
emb_path = os.path.join(BASE, "public", "embeddings.bin")
raw = np.frombuffer(open(emb_path, "rb").read(), dtype=np.float32)
N_old = len(data)
DIM   = len(raw) // N_old
emb   = raw.reshape(N_old, DIM)
filtered_emb = emb[keep]
filtered_emb.tofile(emb_path)
mb2 = os.path.getsize(emb_path) / 1024 / 1024
print(f"✅  embeddings.bin → {mb2:.1f} MB  ({len(keep):,} × {DIM}d)")
