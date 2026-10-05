# 2026 upgrade validation

Date: 2026-10-04. Branch: `feat/2026-upgrade`, based on master `d0e40251d9de61b3340b8d0d7d83150669f1885a`. Changes are local and uncommitted.

## Implemented behavior

- Default model: Qwen3-1.7B; optional larger model: Qwen3-4B-Instruct-2507. Qwen2.5 selections remain available.
- Native macOS arm64 automatically uses MLX LoRA. Other platforms retain PyTorch CUDA/CPU; CPU does not select MPS. Web training is SFT only.
- Both training backends use native model templates, assistant-only completion loss, and Qwen3 nonthinking responses. LoRA uses rank 8, alpha 16 / MLX scale 2 and dropout 0.1.
- Facts are supplied through bounded reference memories. Style/preference synthesis carries the same context format; old unmarked factual QA is excluded from the new merged dataset.
- New uploads index immediately; document content updates replace chunks/index entries; deletion removes that document's vectors. Model/endpoint changes create isolated embedding collections, retaining old collections for migration.
- llama.cpp server and converter use the same pinned revision, b6500 (`a7a98e0fffed794396b3fbad4dcdbbc184963645`). Metal/CUDA offload depends on both the actual build and platform. Management is restricted to this worktree executable and configured port.

## Automated and integration checks

`./.venv/bin/python -m pytest -q -p no:cacheprovider tests`: **73 passed** (rerun 2026-10-05). This includes native tokenizer/completion masks for both Qwen3 models, MLX source-group splitting and remainder batches, persisted parameters/backend preflight, cancellation/failure contracts, local-server ownership/offload/context budgeting, memory prompt/retrieval/SQLite lifecycle and a real isolated Chroma worker.

GraphRAG runtime/performance changes: stable per-note input wrappers, a runtime settings copy with `${GRAPHRAG_*_API_KEY}` placeholders (keys only in the child environment, tracked `settings.yaml`/`.env` unchanged apart from placeholders), a persistent file cache namespaced by model/endpoint/prompt semantics, configurable concurrency and cache hit/miss metrics. Review fixes: the tracked GraphRAG `.env` was restored, the embedding model default stays `text-embedding-ada-002`, the runner prints the redacted exception type and message to stderr, and the parent logs return code plus stdout/stderr tails on failure (stderr tail as a warning on success). These are covered only by offline tests with mocked HTTP/subprocesses; a full cold/warm GraphRAG index on real user notes, actual speedup and the failure logging path in a real run are not yet validated.

The Chroma worker uses actual Chroma 0.4.24, SQLite, upload/document services and chunking, with deterministic synthetic embeddings and blocked network. It verifies multiple hits, upload/update/delete, other-document retention, same-dimension model and endpoint changes, 1024→1536 changes and legacy collection retention. It does not call Cloudflare.

Poetry lock validation and installation succeeded in the worktree `.venv`. `uv pip check --python .venv/bin/python`: **247 installed packages compatible**. The tested stack pins Transformers 4.53.3, Torch 2.5.1, PEFT 0.16.0, TRL 0.19.1, datasets 3.3.2, MLX 0.29.2 and MLX-LM 0.28.3. The patched local GraphRAG runtime imports successfully. The pinned converter additionally requires mistral-common 1.8.3.

Frontend TypeScript and targeted ESLint checks pass. Production build passes with `npm run build -- --no-lint`. Whole-project ESLint has pre-existing errors in applications, MCP tools, TrainingLog and Header files; those unrelated errors are not repaired here. An initial-state model value was corrected after production prerendering caught an unsafe empty-object cast.

The full Flask application was imported against an isolated temporary SQLite database and explicit safe configuration. `/health` and `/api/trainprocess/training_params` return success; the latter reports `apple_silicon=true`, `mlx_available=true`, `selected_training_backend=mlx`. Fresh application imports still require the existing database bootstrap and registry URL configuration; the smoke test bootstraps the user-config table before eager route imports. No private `.env` or API keys were used in these tests.

## Real M1 Max model run

Hardware: Apple M1 Max, 64 GB unified memory. Nonquantized original HF Qwen3-1.7B weights; batch 1, accumulation 1, sequence limit 2048, checkpointing, two microbatches. The synthetic fixture contains 48 records across independent sources; assistant targets yield 51 train and 4 validation examples.

| Run | Peak MLX memory | Training loss | Validation loss | Purpose |
| --- | --- | --- | --- | --- |
| Initial CLI smoke (dropout 0.0) | 3.904 GB | 11.402 → 8.500 | 7.969 → 7.296 | Initial execution/fusion/conversion check |
| Final Web-service smoke (dropout 0.1) | 3.942 GB | 11.402 → 8.250 | 7.969 → 7.315 | Final parameters and actual Web orchestration |

The final test called the actual `TrainProcessService.train()`, `merge_weights()` and `convert_model()` methods. It substituted only the synthetic fixture path, skipping online data synthesis. Each subprocess returned zero and produced its artifact; each corresponding progress step became completed. Backend/objective metadata persisted as MLX/SFT/DPO-not-executed. The three stages took approximately 35 seconds (23:31:49–23:32:24), including model loading, training, fusion and F16 GGUF conversion. Reported completion-token throughput was 7.58 and 12.22 tokens/s for the two microbatches; this is not total input throughput or a sustained benchmark.

The final fused GGUF was loaded through the actual local service on isolated port 8081. Logs show all 29 layers offloaded to Metal, context 8192, one slot. Both ordinary and streamed answers used the current synthetic reference: the meeting moved from Tuesday to Thursday at 3 PM. No thinking segment appeared in returned content. A no-reference query about the next project meeting returned uncertainty instead of inventing a time. A 102-message synthetic history was trimmed to 14 messages while retaining the current question and reserving 512 output tokens. Tests additionally check the exact template/token budget and preservation of system rules.

The training page was inspected in ego-browser against isolated backend 8003/frontend 3011 using a temporary synthetic identity. It displayed Qwen3-1.7B as recommended, MLX automatic acceleration, SFT-only and nonthinking guidance, and completed Train/Merge/Convert steps. Numeric batch/accumulation/sequence and checkpoint controls are connected to the same request parameters. An end-to-end browser click through cloud synthesis was not performed because it requires user-configured models/keys and real source memories. Temporary test servers were stopped; existing 8002/3010 services were preserved. Synthetic progress was archived alongside the model artifacts so it does not appear as an unfinished user training run.

## Scope limits

- Two training steps establish execution, finite loss and deployment compatibility, not convergence, personality quality or a measured factual-accuracy improvement.
- Qwen3-4B-Instruct-2507 configuration and tokenizer were validated; full 4B weights/training/fusion/inference were not run.
- CUDA hardware and Docker builds were not executed on this Mac; their build/setup paths and backend-selection contracts were checked.
- Real Cloudflare embedding calls and registry/public MCP roundtrips were not executed. Users supply their own Cloudflare/model configuration. Private values are not migrated or printed.
- Real Chroma validates retrieval changes; the Metal examples validate model behavior on synthetic reference prompts. These are separate checks, not a statistical end-to-end memory quality benchmark.
