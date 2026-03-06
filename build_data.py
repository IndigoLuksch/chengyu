"""
build_data.py — rebuild public/data.json from IdiomTranslate30 + translations.json

Usage: python build_data.py
Output: public/data.json  (replaces existing file)

Fields per idiom:
  c  — Chinese characters
  p  — Pinyin
  l  — Literal translation  (from Haiku)
  m  — Meaning              (from Haiku)
  t  — Author span translations (from IdiomTranslate30, for display)
  e  — Embed text: search + meaning + spans  (stripped before deploy by generate.mjs)
"""
from datasets import load_dataset
from collections import defaultdict
from pypinyin import pinyin, Style
import json, os

BASE = os.path.dirname(__file__)

# ── Load IdiomTranslate30 ────────────────────────────────────────────────────
print("Loading IdiomTranslate30…")
ds = load_dataset("kenantang/IdiomTranslate30", split="train")
en = ds.filter(lambda x: x["source_language"] == "Chinese" and x["target_language"] == "English")
print(f"  {len(en):,} English rows, building index…")

# ── Group spans by idiom ──────────────────────────────────────────────────────
creative = defaultdict(list)
analogy  = defaultdict(list)
author   = defaultdict(list)

for row in en:
    ch = row["idiom"]
    for lst, key in [(creative, "span_creatively"), (analogy, "span_analogy"), (author, "span_author")]:
        val = (row[key] or "").strip()
        if val and val not in lst[ch]:
            lst[ch].append(val)

chinese_idioms = sorted(creative.keys())
print(f"  {len(chinese_idioms):,} unique Chinese idioms")

# ── Load Haiku translations ───────────────────────────────────────────────────
trans_path = os.path.join(BASE, "translations.json")
trans_map  = {}
if os.path.exists(trans_path):
    trans = json.load(open(trans_path, encoding="utf-8"))
    trans_map = {t["c"]: t for t in trans if "error" not in t}
    print(f"  {len(trans_map):,} Haiku translations loaded")
else:
    print("  ⚠ translations.json not found — run translate.py first")

# ── Build records ─────────────────────────────────────────────────────────────
def make_pinyin(ch):
    parts = pinyin(ch, style=Style.TONE, heteronym=False)
    return " ".join(p[0] for p in parts)

records = []
for ch in chinese_idioms:
    haiku = trans_map.get(ch, {})

    records.append({
        "c":  ch,
        "p":  make_pinyin(ch),
        "l":  haiku.get("literal", ""),
        "m":  haiku.get("meaning", ""),
        "s":  haiku.get("search", ""),   # kept separate so generate.mjs can weight equally
    })

# ── Write output ──────────────────────────────────────────────────────────────
out = os.path.join(BASE, "public", "data.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, separators=(",", ":"))

mb = os.path.getsize(out) / 1024 / 1024
with_haiku = sum(1 for r in records if r["l"])
print(f"\n✅  public/data.json  ({mb:.1f} MB, {len(records):,} idioms, {with_haiku:,} with Haiku translations)")
print("   Next: node generate.mjs  (re-embed with new data)")
