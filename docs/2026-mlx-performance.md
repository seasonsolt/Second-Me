# MLX training performance evidence

Measured on 2026-10-05 using Apple M1 Max, 64 GB shared memory, native nonquantized Qwen3-1.7B, MLX 0.29.2, MLX-LM 0.28.3 and Transformers 4.53.3. All runs are sequential local SFT with no concurrent inference. Full completion markers and per-step metrics are preserved in [the JSON evidence](2026-mlx-performance.json).

## Results

| Fixture | Batch | Checkpoint | Training wall (s) | MLX peak (GB) | Padding | Updates |
|---|---:|---|---:|---:|---:|---:|
| 83–171 tokens | 1 | On | 29.22 | 4.11 | 0.00% | 51 |
| 83–171 tokens | 4 | Off | 16.33 | 7.88 | 1.47% | 13 |
| 83–171 tokens | 8 | Off | 16.27 | 12.25 | 3.83% | 7 |
| 512–2048 tokens | 1 | On | 30.83 | 7.31 | 0.00% | 8 |
| 512–2048 tokens | 1 | Off | 26.24 | 21.01 | 0.00% | 8 |
| 512–2048 tokens | 2 | Off | 23.54 | 37.02 | 9.09% | 4 |
| 512–2048 tokens | 4 | On | 33.58 | 17.79 | 16.65% | 2 |
| 512–2048 tokens | 4 | Off | Failed on first step | Not reported | — | 0 |

## Interpretation

- Keep batch 1 and gradient checkpointing enabled as the general default. On mixed 512–2048 token sequences it completed at 7.31 GB peak. Batch 4 with checkpointing increased both elapsed time and memory.
- For the short fixture, batch 4 with checkpointing disabled reduced training wall time from 29.22 to 16.33 seconds (44.1%). Batch 8 gave nearly the same time with substantially higher memory, so batch 4 is the measured short-example option.
- On long sequences, checkpointing disabled at batch 1 took 26.24 seconds and 21.01 GB; batch 2 took 23.54 seconds and 37.02 GB. Its roughly 10% improvement over batch 1 costs roughly 76% more peak memory. This is an optional measured tradeoff, not a universal preset.
- Long batch 4 without checkpointing failed with a Metal insufficient-memory error. Every long run used `mx.set_memory_limit(28 * 1024**3)`. The installed MLX API defines this as an allocation guideline, not a hard cap; batch 2 exceeded it. These results do not establish that a 64 GB machine cannot run batch 4 under another configuration. An unrestricted/default-limit run was not tested.

## Method and limits

The short fixture has 51 training and 4 validation examples. The synthetic long fixture has 8 training and 1 validation example with exact 512, 1024 and 2048 token lengths. Source groups are split before batching; training length grouping preserves every example and remainder batch. Validation keeps the source-split order. All runs use one epoch, seed 42, learning rate 0.0001, gradient accumulation 1, maximum length 2048, LoRA rank 8/scale 2/dropout 0.1 across all layers, and a 4 GiB cache limit. Both benchmarks use full validation (`--validation-batches -1`); the Web default of 4 validation batches was not speed-benchmarked here.

The measured wall timer covers the trainer, validation and adapter save, but excludes model/data loading, LoRA fusion, GGUF conversion and serving. Memory is the MLX allocator peak in decimal GB, not system RSS or dedicated GPU VRAM. The JSON also records throughput after the first reported microbatch is omitted. This is only a warm-up approximation: later sequence shapes can still compile, and the long batch-4 run has just one remaining step. Avoid treating it as stable sustained throughput.

All compared cases consume identical examples and assistant target tokens within each fixture. Larger batches have fewer optimizer updates; reported loss differences cannot demonstrate equivalent convergence or model quality. MLX-LM reports its final validation before the last optimizer update. These synthetic one-epoch measurements are performance diagnostics, not quality evaluation.

`input_tokens` counts complete sequences, `effective_input_tokens` counts predictor positions (sequence length minus one), and `supervised_tokens` counts assistant targets. `model_input_tokens` counts nonpadding physical input positions including EOS where shorter rows retain it; `padding_tokens / padded_model_input_tokens` gives the physical padding fraction. The short batch-1 baseline predates the explicit effective-input field; its equivalent count is `input_tokens - processed_examples`.

## Reproduce

Short runs:

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -m lpm_kernel.L2.mlx_training.train \
  --model resources/L2/base_models/Qwen3-1.7B --output run/mlx-performance-20261005/batch4-no-checkpoint \
  --data tests/fixtures/qwen3_training.json --user-name Tester --lr 0.0001 --epochs 1 \
  --batch-size 4 --grad-accumulation 1 --max-length 2048 --seed 42 --validation-batches -1
```

Use batch 1 plus `--grad-checkpoint` for the short baseline, or batch 8 without it for the other short run. Length grouping is enabled by default.

Long runs (substitute batch 1, 2 or 4, add `--grad-checkpoint` for the two checkpoint cases, and use a distinct output folder):

```sh
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python -u -c \
  'import mlx.core as mx; mx.set_memory_limit(28 * 1024**3); from lpm_kernel.L2.mlx_training.train import main; main()' \
  --model resources/L2/base_models/Qwen3-1.7B --output run/mlx-performance-20261005/long-batch1-no-checkpoint \
  --data tests/fixtures/mlx_performance_lengths.json --user-name Tester --lr 0.0001 --epochs 1 \
  --batch-size 1 --grad-accumulation 1 --max-length 2048 --seed 42 --validation-batches -1
```

Raw logs, final adapters and original marker files remain under the ignored `run/mlx-performance-20261005/` directory. The tracked JSON preserves all measured values and fixture hashes without requiring those local files.
