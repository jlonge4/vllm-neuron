# Qwen3-30B-A3B Benchmark Results on trn3.48xlarge

## Best Config (Spec Decode n=6, Single Stream)

### Serving Config
| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-30B-A3B |
| Hardware | trn3.48xlarge (16 NeuronCores) |
| Container | public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0 |
| TP | 8 |
| EP | 8 |
| Spec decode | Eagle3 (zhuyksir/EAGLE3-Qwen3-30B-A3B-DenseHead), n=6 |
| Max model len | 4096 |
| Buckets (tokens) | [512, 4096] |
| Buckets (seqs) | [1, 4, 16] |
| Prefix caching | disabled |

### Workload: 1K input / 1K output, single stream, 20 prompts

#### Throughput
| Metric | Value |
|--------|-------|
| Output throughput | 82.9 tok/s |
| Input throughput | 61.5 tok/s |
| Total throughput | 144.4 tok/s |
| Request throughput | 0.18 req/s |

#### End-to-End Latency
| Percentile | Value |
|-----------|-------|
| Mean | 5,695 ms |
| Median (P50) | 5,008 ms |
| P95 | 13,657 ms |
| P99 | 15,200 ms |

#### Time to First Token (TTFT)
| Percentile | Value |
|-----------|-------|
| Mean | 490 ms |
| Median (P50) | 436 ms |
| P95 | 668 ms |
| P99 | 766 ms |

#### Inter-Token Latency (ITL)
| Percentile | Value |
|-----------|-------|
| Mean | 30.2 ms |
| Median (P50) | 0.02 ms (burst from spec decode) |
| P95 | 109.0 ms |
| P99 | 111.8 ms |
| Max | 173.9 ms |

#### Time Per Output Token (TPOT)
| Percentile | Value |
|-----------|-------|
| Mean | 11.4 ms |
| Median (P50) | 11.6 ms |
| P95 | 17.5 ms |
| P99 | 17.5 ms |

---

## Spec Decode Token Sweep (1K/1K single stream)

| Spec Tokens | Output tok/s | Status |
|-------------|-------------|--------|
| 2 | 78.5 | ✓ |
| 4 | 87.3 | ✓ |
| 6 | 94.2 | ✓ (optimal) |
| 8 | — | OOM during compile |

---

## Throughput Scaling (EP + multi-bucket, no spec decode)

| Config | Out tok/s | In tok/s | TTFT ms | ITL ms |
|--------|-----------|----------|---------|--------|
| short-single (128/128, c=1) | 40.2 | 40.9 | 406 | 18.3 |
| medium-single (512/256, c=1) | 51.6 | 84.5 | 396 | 16.9 |
| long-single (2048/512, c=1) | 56.6 | 212.7 | 678 | 15.5 |
| medium-batch4 (512/256, c=4) | 128.1 | 272.5 | 485 | 26.5 |
| medium-batch16 (512/256, c=16) | 251.4 | 646.5 | 769 | 44.9 |
| medium-batch32 (512/256, c=32) | 395.3 | 952.8 | 1,285 | 53.7 |
| long-batch32 (2048/512, c=32) | 353.0 | 1,460.0 | 4,998 | 56.2 |

---

## Artificial Analysis Comparison (10K input, 1500 output)

| Metric | AA Reported | Us (BF16) | Us (Spec n=6) | Gap |
|--------|------------|-----------|---------------|-----|
| Single-stream output speed | 146.3 tok/s | 53.0 tok/s | ~94 tok/s* | 1.55x |

*Extrapolated from 1K/1K results; 10K context requires 16K bucket which OOMs with spec decode.

---

## Known Issues & Next Steps

1. **Prefix caching disabled** — easy win, should improve TTFT and throughput
2. **FP8 quantization** — code written but torch.compile branching issue prevents deployment; would ~2x decode speed
3. **16K context + spec decode** — OOMs during compile; needs memory optimization or smaller draft model
4. **EP weight loading bug in original PR** — fixed on this branch (per-expert checkpoint slicing)
5. **QK-norm shape bug in original PR** — fixed on this branch (.unsqueeze(0) on prefill path)

## Branch
`jlonge4/qwen3-moe-perf` on fork (https://github.com/jlonge4/vllm-neuron)
