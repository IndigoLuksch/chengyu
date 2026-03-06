/**
 * test_quiz_scores.mjs — Simulate quiz grading to calibrate thresholds.
 *
 * Uses the same model + "query: " prefix as the browser quiz does.
 * Tests three categories:
 *   1. CORRECT  — the actual meaning or very close paraphrase
 *   2. PARTIAL  — related but imprecise answers
 *   3. WRONG    — completely unrelated answers
 *
 * Usage: node test_quiz_scores.mjs
 */

import { readFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(readFileSync(path.join(__dirname, "public", "data.json"), "utf8"));
const withMeaning = data.filter(d => d.m);

console.log(`Loaded ${withMeaning.length} idioms with meanings\n`);

// ── Load model ──────────────────────────────────────────────────────────────
console.log("Loading model…");
const { pipeline, env } = await import("@huggingface/transformers");
env.allowRemoteModels = true;
const extractor = await pipeline("feature-extraction", "Xenova/bge-small-en-v1.5");
console.log("Model ready.\n");

// ── Helper ──────────────────────────────────────────────────────────────────
async function cosineSim(textA, textB) {
  const [a, b] = await Promise.all([
    extractor("query: " + textA, { pooling: "mean", normalize: true }),
    extractor("query: " + textB, { pooling: "mean", normalize: true }),
  ]);
  let sim = 0;
  for (let i = 0; i < a.data.length; i++) sim += a.data[i] * b.data[i];
  return sim;
}

// ── Pick 20 random idioms for testing ───────────────────────────────────────
const rng = (n) => Math.floor(Math.random() * n);
const sample = [];
const used = new Set();
while (sample.length < 20) {
  const i = rng(withMeaning.length);
  if (!used.has(i)) { used.add(i); sample.push(withMeaning[i]); }
}

// ── Generate test cases ────────────────────────────────────────────────────
// For each idiom, test:
//   CORRECT:  the actual meaning verbatim
//   CORRECT2: a short paraphrase (first ~8 words of meaning)
//   PARTIAL:  a vaguely related short answer
//   WRONG:    a completely unrelated answer

const wrongAnswers = [
  "eating breakfast in the morning",
  "the weather is nice today",
  "buying groceries at the store",
  "watching television at night",
  "driving a car to work",
  "the cat sat on the mat",
  "cooking dinner for the family",
  "reading a book before bed",
  "walking the dog in the park",
  "cleaning the house on weekend",
  "swimming in the ocean",
  "playing basketball with friends",
  "painting a picture of flowers",
  "doing laundry on Sunday",
  "fixing a broken window",
  "watering the garden plants",
  "taking photos of sunset",
  "listening to music on radio",
  "writing a letter to grandma",
  "feeding fish in the aquarium",
];

function shortParaphrase(meaning) {
  // Take first sentence, truncate to ~6-8 words
  const words = meaning.split(/[.;,]/)[0].split(/\s+/).slice(0, 7);
  return words.join(" ");
}

function partialAnswer(meaning) {
  // Extract a couple of key words to make a vague answer
  const words = meaning.split(/\s+/).filter(w => w.length > 4);
  if (words.length >= 3) return words.slice(0, 2).join(" ");
  return words[0] || "something";
}

// ── Run tests ──────────────────────────────────────────────────────────────
const results = { correct: [], correct2: [], partial: [], wrong: [] };

console.log("Running tests…\n");
console.log("=" .repeat(90));

for (let i = 0; i < sample.length; i++) {
  const idiom = sample[i];
  const meaning = idiom.m;

  // 1. Exact meaning (verbatim)
  const simExact = await cosineSim(meaning, meaning);
  results.correct.push(simExact);

  // 2. Short paraphrase
  const para = shortParaphrase(meaning);
  const simPara = await cosineSim(para, meaning);
  results.correct2.push(simPara);

  // 3. Partial (key words only)
  const part = partialAnswer(meaning);
  const simPartial = await cosineSim(part, meaning);
  results.partial.push(simPartial);

  // 4. Wrong answer
  const simWrong = await cosineSim(wrongAnswers[i], meaning);
  results.wrong.push(simWrong);

  console.log(`${idiom.c}  "${meaning.slice(0, 50)}…"`);
  console.log(`  EXACT    : ${simExact.toFixed(4)}  (verbatim meaning)`);
  console.log(`  PARAPH   : ${simPara.toFixed(4)}  ("${para}")`);
  console.log(`  PARTIAL  : ${simPartial.toFixed(4)}  ("${part}")`);
  console.log(`  WRONG    : ${simWrong.toFixed(4)}  ("${wrongAnswers[i]}")`);
  console.log();
}

// ── Summary stats ──────────────────────────────────────────────────────────
function stats(arr) {
  arr.sort((a, b) => a - b);
  const min = arr[0], max = arr[arr.length - 1];
  const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
  const med = arr[Math.floor(arr.length / 2)];
  return { min: min.toFixed(3), max: max.toFixed(3), mean: mean.toFixed(3), med: med.toFixed(3) };
}

console.log("=" .repeat(90));
console.log("\nSUMMARY STATISTICS:");
console.log("-".repeat(60));
for (const [cat, arr] of Object.entries(results)) {
  const s = stats([...arr]);
  console.log(`  ${cat.padEnd(10)} │ min=${s.min}  med=${s.med}  mean=${s.mean}  max=${s.max}`);
}

console.log("\n\nRECOMMENDED THRESHOLDS:");
console.log("-".repeat(60));

// Excellent threshold: should separate CORRECT2 (paraphrase) from PARTIAL
// Almost threshold: should separate PARTIAL from WRONG
const allParaphrase = [...results.correct2].sort((a, b) => a - b);
const allPartial = [...results.partial].sort((a, b) => a - b);
const allWrong = [...results.wrong].sort((a, b) => a - b);

const p10Para = allParaphrase[Math.floor(allParaphrase.length * 0.1)];
const p90Partial = allPartial[Math.floor(allPartial.length * 0.9)];
const p90Wrong = allWrong[Math.floor(allWrong.length * 0.9)];
const p10Partial = allPartial[Math.floor(allPartial.length * 0.1)];

console.log(`  Paraphrase p10 = ${p10Para.toFixed(3)}   (Excellent threshold should be ≤ this)`);
console.log(`  Partial p90    = ${p90Partial.toFixed(3)}   (Excellent threshold should be ≥ this)`);
console.log(`  Partial p10    = ${p10Partial.toFixed(3)}   (Almost threshold should be ≤ this)`);
console.log(`  Wrong p90      = ${p90Wrong.toFixed(3)}   (Almost threshold should be ≥ this)`);
console.log();

const excellentThresh = ((p10Para + p90Partial) / 2).toFixed(2);
const almostThresh = ((p10Partial + p90Wrong) / 2).toFixed(2);

console.log(`  ★★★ Excellent! threshold = ${excellentThresh}`);
console.log(`  ★★☆ Almost!    threshold = ${almostThresh}`);
console.log(`  ★☆☆ Not quite! = below ${almostThresh}`);
