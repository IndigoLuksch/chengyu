"""
translate.py — translate Chinese idioms using Anthropic API (Haiku)

Usage:
  python translate.py --test 5          # test with 5 idioms, print results
  python translate.py --test 3 --prompt prompt.txt  # test with custom prompt
  python translate.py                   # run full translation (resumes if interrupted)
  python translate.py --model claude-sonnet-4-5-20250929  # use a different model

Output:  translations.json  (array of {c, literal, meaning})
Resume:  skips already-translated idioms automatically

Requires:  pip install anthropic
"""

import anthropic
import json
import os
import sys
import time
import argparse
from pathlib import Path

BASE = Path(__file__).parent

# ── Default prompt ───────────────────────────────────────────────────────────-
DEFAULT_PROMPT = """\
You are a Chinese language expert. Given a Chinese idiom (成语) and some existing English translations for context, provide:
python t
1. **literal**: A natural English phrase that captures what the characters literally say. Grammatically correct in english, but a direct literal translation of the Chinese characters. Provide the single best translation with e.g. do not use slashes.
2. **meaning**: A clear, single sentence, concise explanation of the message that the idiom is trying to convey.
3. **search**: 2-3 plain English sentences describing everyday situations where you would use this idiom. Write the way a normal person would describe the situation, e.g. "When someone keeps making the same mistake over and over. When a friend refuses to learn from experience."

Respond with ONLY valid JSON, no markdown fences:
{{"literal": "...", "meaning": "...", "search": "..."}}

Idiom: {idiom}
Existing translations: {translations}"""

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Translate Chinese idioms via Anthropic API")
parser.add_argument("--test", type=int, metavar="N",
                    help="Test mode: translate N idioms and print results (no save)")
parser.add_argument("--prompt", type=str, metavar="FILE",
                    help="Path to a custom prompt file (use {idiom} as placeholder)")
parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001",
                    help="Model to use (default: claude-haiku-4-5-20251001)")
parser.add_argument("--batch-size", type=int, default=5,
                    help="Number of concurrent requests (default: 5)")
parser.add_argument("--delay", type=float, default=0.1,
                    help="Delay between batches in seconds (default: 0.1)")
args = parser.parse_args()

# ── Load prompt ───────────────────────────────────────────────────────────────
if args.prompt:
    prompt_template = Path(args.prompt).read_text(encoding="utf-8").strip()
    print(f"Using custom prompt from {args.prompt}")
else:
    prompt_template = DEFAULT_PROMPT
    print("Using default prompt (override with --prompt FILE)")

# ── Load data ─────────────────────────────────────────────────────────────────
data = json.loads((BASE / "public" / "data.json").read_text(encoding="utf-8"))
print(f"Loaded {len(data):,} idioms from data.json")

# ── Load existing translations (for resume) ───────────────────────────────────
trans_path = BASE / "translations.json"
existing = {}
if trans_path.exists() and not args.test:
    trans = json.loads(trans_path.read_text(encoding="utf-8"))
    existing = {t["c"]: t for t in trans}
    print(f"Resuming: {len(existing):,} already translated")

# ── Select idioms to translate ────────────────────────────────────────────────
if args.test:
    todo = data[:args.test]
    print(f"\n{'='*60}")
    print(f"TEST MODE: translating {len(todo)} idioms with {args.model}")
    print(f"{'='*60}\n")
else:
    todo = [d for d in data if d["c"] not in existing]
    print(f"Remaining: {len(todo):,} to translate with {args.model}")
    if not todo:
        print("Nothing to do!")
        sys.exit(0)

# ── API client ────────────────────────────────────────────────────────────────
client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

# Build lookup for existing translations
idiom_translations = {d["c"]: d.get("t", []) for d in data}

def translate_one(idiom_text):
    """Call the API for a single idiom, return parsed dict or error. Retries on rate limit."""
    trans = idiom_translations.get(idiom_text, [])
    prompt = prompt_template.replace("{idiom}", idiom_text).replace("{translations}", "; ".join(trans) if trans else "none")
    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=args.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
            return {
                "c": idiom_text,
                "literal": result["literal"],
                "meaning": result["meaning"],
                "search": result.get("search", ""),
            }
        except json.JSONDecodeError:
            return {"c": idiom_text, "error": f"JSON parse error: {raw[:200]}"}
        except Exception as e:
            msg = str(e)
            if "rate_limit" in msg.lower() or "429" in msg or "overloaded" in msg.lower():
                wait = 60 * (attempt + 1)  # 60s, 120s, 180s...
                sys.stdout.write(f"\n  Rate limited — waiting {wait}s (attempt {attempt+1}/5)...\n")
                sys.stdout.flush()
                time.sleep(wait)
            else:
                return {"c": idiom_text, "error": msg}
    return {"c": idiom_text, "error": "Max retries exceeded"}

# ── Run translations ──────────────────────────────────────────────────────────
import concurrent.futures

results = []
errors  = []
t0 = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as pool:
    futures = {pool.submit(translate_one, d["c"]): d["c"] for d in todo}
    done_count = 0

    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        done_count += 1

        if "error" in result:
            errors.append(result)
            symbol = "✗"
        else:
            results.append(result)
            symbol = "✓"

        # Progress
        elapsed = time.time() - t0
        rate = done_count / elapsed if elapsed > 0 else 0
        eta = (len(todo) - done_count) / rate if rate > 0 else 0

        if args.test:
            # In test mode, print full results
            print(f"[{done_count}/{len(todo)}] {result['c']}")
            if "error" in result:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  Literal: {result['literal']}")
                print(f"  Meaning: {result['meaning']}")
                print(f"  Search:  {result.get('search', '')}")
            print()
        else:
            sys.stdout.write(
                f"\r  {symbol} {done_count:,}/{len(todo):,}  "
                f"({done_count/len(todo)*100:.0f}%)  "
                f"{rate:.1f}/s  ETA {eta:.0f}s  "
                f"errors={len(errors)}   "
            )
            sys.stdout.flush()

            # Save progress every 50 translations
            if done_count % 50 == 0:
                all_trans = list(existing.values()) + results
                trans_path.write_text(
                    json.dumps(all_trans, ensure_ascii=False, indent=None),
                    encoding="utf-8",
                )

if not args.test:
    print()

# ── Save final results ────────────────────────────────────────────────────────
if not args.test:
    all_trans = list(existing.values()) + results
    trans_path.write_text(
        json.dumps(all_trans, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    mb = trans_path.stat().st_size / 1024 / 1024
    print(f"\n✅  translations.json  ({mb:.1f} MB, {len(all_trans):,} entries)")

# ── Summary ───────────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print(f"Time: {elapsed:.1f}s  |  Success: {len(results):,}  |  Errors: {len(errors):,}")

if errors:
    print(f"\nFailed idioms:")
    for e in errors[:10]:
        print(f"  {e['c']}: {e['error'][:100]}")
    if len(errors) > 10:
        print(f"  ... and {len(errors)-10} more")

if args.test:
    print(f"\n{'='*60}")
    print("To adjust the prompt, create a text file and run:")
    print(f"  python translate.py --test 5 --prompt my_prompt.txt")
    print(f"\nTo run the full translation:")
    print(f"  python translate.py")
    print(f"{'='*60}")
