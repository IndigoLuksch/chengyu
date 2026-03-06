"""
analyse_dataset.py — run locally to understand IdiomTranslate30
Usage: python analyse_dataset.py
"""
from datasets import load_dataset
from collections import Counter
import json, os

ds = load_dataset("kenantang/IdiomTranslate30", split="train")

# ── English rows only ─────────────────────────────────────────────────────────
en = ds.filter(lambda x: x["target_language"] == "English")
print(f"English rows:    {len(en):,}")

idioms_in_dataset = set(en["idiom"])
print(f"Unique idioms:   {len(idioms_in_dataset):,}")

# ── Compare with existing data.json ──────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), "public", "data.json")
existing  = json.load(open(data_path))
existing_chinese = set(d["c"] for d in existing)
print(f"Existing idioms: {len(existing_chinese):,}")
print(f"Overlap:         {len(idioms_in_dataset & existing_chinese):,}")
print(f"New only:        {len(idioms_in_dataset - existing_chinese):,}")

# ── Sentences per idiom ───────────────────────────────────────────────────────
counts = Counter(en["idiom"])
vals   = list(counts.values())
print(f"\nSentences/idiom: min={min(vals)}  max={max(vals)}  avg={sum(vals)/len(vals):.1f}")

# ── Sample spans for one idiom ────────────────────────────────────────────────
sample = "威逼利诱"
rows   = [r for r in en if r["idiom"] == sample]
print(f"\nSample spans for {sample} ({len(rows)} rows):")
for r in rows[:2]:
    print(f"  creative: {r['span_creatively']}")
    print(f"  analogy:  {r['span_analogy']}")
    print(f"  author:   {r['span_author']}")
