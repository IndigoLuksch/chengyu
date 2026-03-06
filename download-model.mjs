/**
 * download-model.mjs
 * Downloads the bge-small-en-v1.5 ONNX model files into public/models/
 * so they're self-hosted on Cloudflare — no runtime HuggingFace dependency.
 *
 * Usage:  node download-model.mjs
 * Output: public/models/Xenova/bge-small-en-v1.5/  (~43 MB total)
 */

import { mkdirSync, writeFileSync, statSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const BASE   = "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main";
const OUTDIR = path.join(__dirname, "public", "models", "Xenova", "bge-small-en-v1.5");

// Small files go into public/models/ (served by Cloudflare Pages).
// The ONNX (~32 MB) exceeds Cloudflare Pages' 25 MB limit — upload it to R2 manually.
const FILES = [
  "config.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  // onnx/model_quantized.onnx → upload to Cloudflare R2 instead
];

mkdirSync(OUTDIR, { recursive: true });

async function download(file) {
  const url  = `${BASE}/${file}`;
  const dest = path.join(OUTDIR, file);
  process.stdout.write(`  ${file} … `);

  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);

  const buf = Buffer.from(await res.arrayBuffer());
  writeFileSync(dest, buf);

  const kb = statSync(dest).size / 1024;
  console.log(`${kb.toFixed(0)} KB`);
}

console.log(`Downloading model files to ${OUTDIR}\n`);
for (const file of FILES) await download(file);

console.log(`\n✅  Done. Commit the public/models/ folder and redeploy.`);
