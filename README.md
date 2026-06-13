# Optimize-LLM-Inference-with-FP16-INT8-and-INT4-Quantization-using-BitsAndBytes

---

## Overview

This project benchmarks `facebook/opt-2.7b` across three numerical precisions — **FP16**, **INT8**, and **NF4 (4-bit)** — to analyze trade-offs between inference speed, GPU memory usage, and model quality. The goal is to develop a practical decision-making framework for deploying LLMs in resource-constrained production environments.

---

## Project Structure
├── llm_quantization_benchmark.ipynb

├── requirements.txt

└── results/

├── summary.json

├── quantization_tradeoffs.png

└── throughput_comparison.png

---

## Quantization Methods

| Method | Bits | Description |
|--------|------|-------------|
| FP16 | 16-bit | Half-precision baseline, de-facto standard for LLM inference |
| INT8 | 8-bit | BitsAndBytes LLM.int8() — lossy integer quantization |
| NF4 | 4-bit | NormalFloat-4 from QLoRA paper — optimal for normally-distributed weights |

---

## Benchmark Results

| Metric | FP16 | INT8 | NF4 |
|--------|------|------|-----|
| GPU Memory (MB) | 9229.43 | 9229.50 | 9229.50 |
| Throughput batch=1 (tok/s) | 39.22 | 6.90 | 17.69 |
| Throughput batch=4 (tok/s) | 139.71 | 24.79 | 48.28 |
| Throughput batch=16 (tok/s) | 436.03 | 88.52 | 189.23 |
| First-Token Latency (ms) ↓ | 29.13 | 189.84 | 78.83 |
| WikiText-2 Perplexity ↓ | 13.0446 | 13.0485 | 13.3017 |
| HellaSwag Accuracy ↑ | 0.4300 | 0.4300 | 0.4350 |

---

## Key Findings

- **FP16** delivers the best throughput and lowest latency — the gold standard for quality and speed
- **INT8** preserves model quality (perplexity within 0.004 of FP16) but is slower at small batch sizes due to quantization overhead on T4 GPU
- **NF4** achieves competitive quality with meaningful memory savings — best choice for memory-constrained deployments

---

## Deployment Decision Guide

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| Real-Time Chatbot (≤8GB VRAM) | **NF4** | Lowest memory, acceptable latency |
| Offline Batch Summarization | **FP16** | Best throughput at large batch sizes |
| Sensitive Medical Q&A | **FP16** | Lowest perplexity, no quantization noise |

---

## Setup

```bash
pip install -r requirements.txt
```

1. Open `llm_quantization_benchmark.ipynb` in Google Colab
2. Set runtime to **T4 GPU** — Runtime → Change runtime type
3. Run all cells — Runtime → Run all
4. Results saved automatically to `results/summary.json`

---

## Requirements
transformers>=4.32.1

bitsandbytes>=0.41.1

accelerate>=0.22.0

torch

datasets>=2.14.0

scipy

numpy

matplotlib

---

## Tools and Libraries

| Tool | Purpose |
|------|---------|
| `transformers` | Load and run `facebook/opt-2.7b` |
| `bitsandbytes` | INT8 and NF4 quantization kernels |
| `accelerate` | Automatic device mapping |
| `datasets` | WikiText-2 and HellaSwag evaluation |
| `torch.cuda.Event` | Accurate GPU-side timing |
| `matplotlib` | Benchmark result visualization |

---
