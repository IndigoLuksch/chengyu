#!/usr/bin/env python3
"""
Pre-generate audio files for all chengyu using Microsoft Edge TTS (free, neural quality).
Outputs MP3 files to public/audio/{chinese}.mp3
"""

import asyncio
import json
import os
import sys
import edge_tts

VOICE = "zh-CN-XiaoxiaoNeural"  # High quality female Mandarin voice
OUTPUT_DIR = "public/audio"
DATA_FILE = "public/data.json"

async def generate(text, path):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(path)

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(DATA_FILE) as f:
        data = json.load(f)

    total = len(data)
    skipped = 0
    generated = 0
    errors = 0

    for i, idiom in enumerate(data):
        chinese = idiom["c"]
        # Filename: use the Chinese characters directly (they're valid in filenames)
        filename = os.path.join(OUTPUT_DIR, f"{chinese}.mp3")

        if os.path.exists(filename):
            skipped += 1
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{total}] {skipped} skipped, {generated} generated, {errors} errors")
            continue

        try:
            await generate(chinese, filename)
            generated += 1
        except Exception as e:
            print(f"  ERROR on {chinese}: {e}")
            errors += 1

        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"[{i+1}/{total}] {skipped} skipped, {generated} generated, {errors} errors")

    print(f"\nDone. {generated} new files generated, {skipped} already existed, {errors} errors.")
    print(f"Audio files saved to {OUTPUT_DIR}/")

asyncio.run(main())
