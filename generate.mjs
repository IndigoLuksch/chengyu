/**
 * generate.mjs — run once to build public/embeddings.bin
 *
 * Usage:
 *   node generate.mjs
 *
 * Output:
 *   public/embeddings.bin  (~6.3 MB, Float32, shape [4310 × 384])
 *
 * Requires Node 18+ and internet access to download the model once.
 * The model (~45 MB) is cached in ~/.cache/huggingface after the first run.
 */

import { writeFileSync, readFileSync, statSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── Load data ─────────────────────────────────────────────────────────────────
const data = JSON.parse(
  readFileSync(path.join(__dirname, "public", "data.json"), "utf8")
);
console.log(`Loaded ${data.length} idioms from public/data.json`);

// BGE models require "passage: " prefix on documents at index time.
// We embed `s` (search) and `m` (meaning) separately, then average + renormalise
// so both fields receive equal weight regardless of text length.
const searchTexts  = data.map(d => "passage: " + (d.s || d.m));
const meaningTexts = data.map(d => "passage: " + d.m);

// ── Load model ────────────────────────────────────────────────────────────────
console.log("Loading model (downloads ~23 MB on first run, then cached)…");
const { pipeline, env } = await import("@huggingface/transformers");
env.allowRemoteModels = true;

const extractor = await pipeline(
  "feature-extraction",
  "Xenova/bge-small-en-v1.5"
);

// ── Embed helper ──────────────────────────────────────────────────────────────
async function embedAll(texts, label) {
  const BATCH = 64;
  const N     = texts.length;
  let   DIM   = null;
  const vecs  = [];
  const t0    = Date.now();
  for (let i = 0; i < N; i += BATCH) {
    const batch  = texts.slice(i, i + BATCH);
    const output = await extractor(batch, { pooling: "mean", normalize: true });
    vecs.push(...Array.from(output.data));
    if (!DIM) DIM = output.data.length / batch.length;
    const done = Math.min(i + BATCH, N);
    process.stdout.write(`\r  [${label}] ${done}/${N}  (${Math.round(done/N*100)}%)   `);
  }
  console.log(`\n  Done in ${((Date.now()-t0)/1000).toFixed(1)}s`);
  return { vecs, DIM };
}

console.log(`\nEmbedding search fields (${data.length} idioms)…`);
const { vecs: searchVecs, DIM } = await embedAll(searchTexts, "search");

console.log(`Embedding meaning fields (${data.length} idioms)…`);
const { vecs: meaningVecs } = await embedAll(meaningTexts, "meaning");

// ── Average + renormalise ────────────────────────────────────────────────────
console.log("Averaging and normalising vectors…");
const allVecs = [];
for (let i = 0; i < data.length; i++) {
  const off = i * DIM;
  // Average the two vectors
  const avg = new Float32Array(DIM);
  for (let d = 0; d < DIM; d++) avg[d] = (searchVecs[off+d] + meaningVecs[off+d]) / 2;
  // Renormalise to unit length
  let norm = 0;
  for (let d = 0; d < DIM; d++) norm += avg[d] * avg[d];
  norm = Math.sqrt(norm);
  for (let d = 0; d < DIM; d++) allVecs.push(avg[d] / norm);
}

console.log(`Embedding dimension: ${DIM}`);

// ── Write binary ──────────────────────────────────────────────────────────────
const f32  = new Float32Array(allVecs);
const buf  = Buffer.from(f32.buffer);
const dest = path.join(__dirname, "public", "embeddings.bin");

writeFileSync(dest, buf);
const mb = statSync(dest).size / 1024 / 1024;
console.log(`\n✅  public/embeddings.bin  (${mb.toFixed(1)} MB)`);

// Write data.json — s (search) field kept for quiz grading in browser
const dataPath = path.join(__dirname, "public", "data.json");
writeFileSync(dataPath, JSON.stringify(data));
const dataMb = statSync(dataPath).size / 1024 / 1024;
console.log(`✅  public/data.json stripped of embed text  (${dataMb.toFixed(1)} MB)`);
console.log("   Deploy the public/ folder to Cloudflare Pages.");
