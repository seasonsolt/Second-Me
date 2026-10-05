# Apple Silicon MLX training

The Web training workflow selects MLX automatically on native macOS arm64. Linux and other platforms use the existing PyTorch CUDA/CPU path. Docker on a Mac runs Linux and does not provide native MLX or Metal training.

The default model is `Qwen/Qwen3-1.7B`. The larger option is `Qwen/Qwen3-4B-Instruct-2507`. Existing Qwen2.5 choices remain available. Each model uses its own native chat template; Qwen3 runs without thinking output. The Web workflow performs **SFT only**, and records `training_objective=sft` and `dpo_executed=false`.

## Setup and Web training

Run setup from the repository root with an arm64 Python 3.12 and an isolated project virtual environment. The checked-in Poetry lock fixes the tested Transformers, PEFT, TRL, MLX and MLX-LM versions. Setup installs the patched local GraphRAG package after resolving its dependencies and builds the llama.cpp revision from `dependencies/llama.cpp.version`.

On the training screen, the default **Auto** backend selects MLX on Apple Silicon; if its dependencies are missing, training reports the error instead of silently falling back to CPU. The initial settings are batch size 1, gradient accumulation 1, maximum sequence length 2048 and gradient checkpointing enabled. Increase these only after measuring your model's memory use. The 4B option has not been fully trained in this upgrade's local acceptance run.

Training reads `resources/L2/data/merged.json`. Samples carry style/preferences and bounded reference context. Facts should come from retrieval; factual QA without the new memory-format marker is excluded when assembling the new dataset. Train/validation splitting keeps rewrites and turns sharing a source, document or context together. Only assistant completion tokens contribute to loss, and oversized samples fail preflight instead of being silently cut.

Outputs follow the shared Web pipeline:

1. LoRA adapter: `resources/model/output/personal_model/<model>/`.
2. `mlx_lm fuse` produces HF weights in `resources/model/output/merged_model/<model>/`.
3. The pinned converter creates `resources/model/output/gguf/<model>/model.gguf`.
4. The application's local service starts llama-server with Metal offload and the native nonthinking template.

A failed or cancelled training run does not report successful completion. Fusion requires a completion marker identifying the matching base model. Cancellation targets the workflow's child process group.

## CLI smoke run

Download the original HF model into `resources/L2/base_models/Qwen3-1.7B`, then run from the repository root:

```bash
.venv/bin/python -m lpm_kernel.L2.mlx_training.train \
  --model resources/L2/base_models/Qwen3-1.7B \
  --output resources/model/output/personal_model/Qwen3-1.7B \
  --data tests/fixtures/qwen3_training.json \
  --epochs 1 --max-steps 2 --max-length 2048 --grad-checkpoint

.venv/bin/python -m mlx_lm fuse \
  --model resources/L2/base_models/Qwen3-1.7B \
  --adapter-path resources/model/output/personal_model/Qwen3-1.7B \
  --save-path resources/model/output/merged_model/Qwen3-1.7B

mkdir -p resources/model/output/gguf/Qwen3-1.7B
.venv/bin/python lpm_kernel/L2/convert_hf_to_gguf.py \
  resources/model/output/merged_model/Qwen3-1.7B \
  --outfile resources/model/output/gguf/Qwen3-1.7B/model.gguf --outtype f16
```

`max_steps` counts MLX microbatches and must be divisible by gradient accumulation. The shell wrappers accept `PYTHON_EXECUTABLE`; they use the same training/conversion modules as the Web workflow.

## Local acceptance evidence (2026-10-04)

On an M1 Max with 64 GB unified memory, the nonquantized Qwen3-1.7B two-step synthetic smoke run processed 51 training targets and 4 validation targets. MLX reported peak memory **3.942 GB**, training loss **11.402 → 8.250** and validation loss **7.969 → 7.315**. This verifies execution and finite loss, not personality quality or convergence.

Fusion and the full F16 GGUF conversion succeeded. llama-server loaded all 29 layers onto Metal with an 8192-token context and one slot. Regular and streamed completions used an updated reference fact; oversized history was trimmed while preserving the current question. No thinking segment appeared in the returned content. See `docs/2026-upgrade-validation.md` for the wider acceptance record and remaining untested configurations.

## Performance controls and measured tradeoffs

MLX groups training samples by length and randomizes the resulting batches to reduce padding without dropping remainder examples. Disable this with `--no-group-by-length` when comparing order effects. Validation keeps the original split order. The Web default evaluates up to four validation batches; use `--validation-batches -1` for full validation. Both controls are persisted with the training parameters.

The [2026-10-05 performance report](../../../docs/2026-mlx-performance.md) and [machine-readable evidence](../../../docs/2026-mlx-performance.json) compare short examples and mixed 512–2048 token sequences on Qwen3-1.7B. Batch 4 without checkpointing was faster on the short fixture; long sequences had much higher memory use, so the Web defaults remain batch 1 with checkpointing enabled. The reports include timer scope, padding, allocator-memory limitations and quality caveats.
