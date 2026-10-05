# Second-Me 2026

English | [中文](README_zh.md)

A **community fork** of [mindverse/Second-Me](https://github.com/mindverse/Second-Me). Not an official release and not affiliated with Mindverse.

Upstream's last substantive update was May 2025. Since then, small models, on-device fine-tuning and agent memory have all moved fast. This fork brings Second-Me in line with 2026 practice:
- Qwen3 base models.
- Native MLX training on Apple Silicon.
- A redesign in which **facts live in a retrieval memory layer and the LoRA only learns to sound like you**.

The roadmap below is the backbone of this fork.

> Status: experimental. Phase 1 is done and benchmarked against upstream (see below). Later phases are ordered by priority.

## Roadmap

### Phase 1: Catch up the foundation (✅ done)

| Industry practice (2026) | Upstream | This fork |
|---|---|---|
| New small-model generation: 1–4B models (Qwen3, Gemma 4) rival last year's 7B | Qwen2.5, default 0.5B | Default Qwen3-1.7B, optional Qwen3-4B-Instruct-2507 |
| On-device fine-tuning is routine (MLX on Mac, Unsloth on NVIDIA) | CUDA/CPU only; on Mac the Trainer silently picks MPS and leaks memory | macOS arm64 auto-selects MLX LoRA; train → fuse → GGUF → serve, all from the Web UI |
| Memory and persona are separate: facts in external memory, weights carry style | Facts are trained into the LoRA, so every new memory means retraining | Facts are retrieved; L2 learns style, preferences and values; adding, editing or deleting a memory needs no retraining |
| Long context with context budgeting | 1024 tokens per request, so retrieved memories don't fit | 8192 context, a token budget for reference memories, and the current question is never truncated |
| Native chat templates with completion-only loss | ChatML, loss on every token | Native template, assistant-only loss, Qwen3 non-thinking by default |

### Phase 2: Make the memory layer production-grade (🔜 next, top priority)

The evaluation shows the memory layer works (78.1% fact accuracy) but is still the bottleneck: one in five facts is missed, the model invents answers after a fact is deleted, and retrieval depends on a per-model threshold.

| Direction | Industry practice | Plan |
|---|---|---|
| **Atomic facts instead of large text chunks** | Mem0-style systems extract standalone facts from conversations and documents and store each one separately | Today's 4,000-character chunks dilute relevance (top-1 similarity is only 0.52–0.68). Extract facts and embed each one on its own |
| **Hybrid retrieval and reranking** | Keyword (BM25) plus dense retrieval, then a reranker, is the standard RAG stack | Replace pure dense search with its hard-coded 0.7 threshold, so the threshold no longer has to be calibrated per embedding model |
| **Temporal memory and conflict resolution** | Temporal knowledge graphs (Zep / Graphiti) keep the newest version of a fact and mark older ones as superseded | Handles updates like "the cat was called Cheese, later renamed Mochi", reusing the GraphRAG entity graph the project already builds |
| **Learn to say "I don't know"** | Train with "no retrieval hit, so abstain" negatives so the model knows the limits of what it knows | Fix the made-up answers after deletion (currently 0/2) |
| **Memory consolidation and forgetting** | Agent memory systems periodically merge, compress and retire old memories | Update L1 summaries incrementally instead of rebuilding them from scratch |

### Phase 3: A more faithful persona (📋 mid-term)

| Direction | Industry practice | Plan |
|---|---|---|
| Style data in the user's own words | Use text the user actually wrote as style samples, not synthetic "ideal" answers | Remove upstream's hard-coded English; extract real expressions from the user's documents as style samples |
| Preference optimization | DPO / SimPO / KTO are standard alignment for small models | Bring back upstream's DPO stage (Mac is SFT-only today) with an MLX implementation |
| Synthetic-data quality control | Score synthetic samples with a judge model and drop weak ones | Automatically score synthesized data and keep only samples that pass |
| Adapter hot-swap | llama.cpp and MLX both load LoRA adapters at runtime | Memory changes need no retraining; style updates only swap the adapter, with no re-merge or re-conversion |

### Phase 4: Join the agent ecosystem (📋 mid-term)

| Direction | Industry practice | Plan |
|---|---|---|
| MCP as the standard tool interface | All major agent clients support MCP | Expose "search my memory" and "answer as me" as MCP tools, building on upstream's MCP server |
| Agent-to-agent protocols | Protocols such as A2A let agents call each other | Replace upstream's network features, which depend on the now-unreachable `app.secondme.io` |
| Fully local | Embeddings and inference run locally, with no cloud dependency | Default to a local bge-m3 (MLX or Ollama) so personal data stays on the machine |

### Phase 5: Efficiency and evaluation (📋 ongoing)

| Direction | Plan |
|---|---|
| Faster training | Larger batches and no gradient checkpointing (peak was only 7.7 GB on M1 Max); expected to cut training time to about a third |
| Faster data synthesis | 🧪 **Early implementation (experimental)**: configurable synthesis concurrency; a GraphRAG runner that counts cache hits and uses a fixed prompt template (a random template choice was why the cache never hit); timing for every stage. The real speedup has not yet been measured with a cold and warm cache on real data. Next: non-reasoning models, targeting about 1.5 h for the full pipeline instead of about 5 h |
| Quantized serving | GGUF Q4_K_M and MLX 4-bit to cut latency (style answers currently take about 18 s at the median) |
| Standard benchmarks | Besides this fork's eval, add long-term-memory benchmarks (LongMemEval, LoCoMo) and persona-consistency evaluation, and run them in CI |
| Validation coverage | Full Qwen3-4B training, an end-to-end run on real CUDA hardware, and separating the effect of a bigger model from the effect of the new design |

## Evaluation: 2025 vs 2026 (what Phase 1 delivered)

Upstream and this fork were compared on the same data (10 documents), the same training parameters, the same 59 questions and the same judge model. Upstream ran as shipped. The harness and method are in [eval/](eval/).

| Metric (retrieval on) | 2025 upstream | 2026 this fork |
|---|---|---|
| Fact accuracy | 4.7% | **78.1%** |
| Doesn't invent unknown facts | 10% | **70%** |
| Persona fidelity (1–5) | 1.0 | **2.2** |
| General answer quality (1–5) | 2.2 | **3.2** |
| Blind A/B (new wins / losses / ties) | — | **52 / 9 / 2** |
| Memory updates without retraining (6 checks: add, edit, delete × 2) | 0 | **3.5** (add 2/2, edit 1.5/2, delete 0/2) |
| Chat-template tag leaks | 0 | 0 |
| Median latency: fact / style questions | 2.2 s / 14.5 s | 4.3 s / 18.3 s |

**Reading the numbers**

- Facts come from the memory layer, not the weights. With retrieval off, the fork answers only 12.5% of fact questions, which is intended: the LoRA learns style, not facts.
- The pre-release review fixes alone took fact accuracy from 40.6% to 78.1% on the **same trained model**: a token-based reference budget instead of a byte-based one, 1,000-character chunks instead of 4,000, and a per-embedding-model threshold. Before those fixes, the default 0.7 threshold retrieved nothing with bge-m3, so the shipped pre-fix code scored 10.9%.
- A good part of the gains in persona and general quality comes from the larger base model (0.5B → 1.7B). This evaluation cannot separate the two effects.
- Still open: the model invents an answer after a fact is deleted (0/2), and persona fidelity is only 2.2/5. Phases 2 and 3 address these.

**Setup**: Apple M1 Max 64 GB. The 2025 model was Qwen2.5-0.5B, trained on an RTX 4070 Laptop (CUDA); the 2026 model was Qwen3-1.7B, trained with MLX on the M1 Max. `gpt-6.1-sol` did the data synthesis and the judging; embeddings came from Cloudflare `@cf/baai/bge-m3`.

## Reliability fixes

Issues you hit when running the full pipeline, all fixed in this fork:

- Embedding requests are batched, since providers cap tokens per request; errors now include the provider's message.
- L0/L1 LLM timeouts raised from 30/45 s to 180 s to suit reasoning models.
- GraphRAG success is judged by exit code, so warnings on stderr no longer count as failures.
- The retrieval threshold is chosen per embedding model (0.5 for bge-m3, otherwise 0.7, which suits OpenAI embeddings) and can be overridden with `L0_SIMILARITY_THRESHOLD` / `L1_SIMILARITY_THRESHOLD`.
- The reference-memory budget is counted in (approximate) tokens, 4096 of them; chunks went from 4,000 to 1,000 characters. Rebuild existing indexes in one call with `POST /api/documents/reindex`.
- Training samples over the length limit are skipped and counted instead of failing the whole run; the default limit is 4096.
- Adding or editing a memory no longer deletes the status biography, and embedding logs no longer include the text.
- Native Linux builds llama.cpp for the GPU (CUDA if `nvcc` exists, otherwise Vulkan) instead of always for the CPU.
- The MLX cache is capped; without a cap it grew to 53 GB on a 64 GB machine.
- Fixed installation of the bundled graphrag package, which is a zip file named `.tar.gz`.

Design documents: [upgrade plan](docs/2026-upgrade-plan.md), [memory layer and lightweight L2](docs/2026-memory-migration.md), [validation log](docs/2026-upgrade-validation.md).

## Quick start (Apple Silicon)

```bash
git clone https://github.com/seasonsolt/Second-Me.git
cd Second-Me
make setup   # needs an arm64 Python 3.12
make start
```

- The training page selects MLX automatically.
- The retrieval threshold is chosen automatically for the embedding model. If you use something other than OpenAI embeddings or bge-m3, calibrate `L0_SIMILARITY_THRESHOLD` yourself.
- If you are upgrading data from upstream, call `POST /api/documents/reindex` first to rebuild indexes with the new chunk size.
- If `python` resolves to an x86 build (`Bad CPU type`), put an arm64 Python 3.12 first on your PATH.

Everything else (Docker deployment, API docs and so on) is the same as upstream; see [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

## Contributing

Issues and PRs are welcome, especially for Phase 2. Please include results from [eval/](eval/) with a change, so the data shows whether it helps.

## Credits and license

- Original project: [mindverse/Second-Me](https://github.com/mindverse/Second-Me); paper: [AI-native Memory 2.0: Second Me](https://arxiv.org/abs/2503.08102).
- License: Apache License 2.0, same as upstream; see [LICENSE](LICENSE). This fork's changes are summarized in [NOTICE](NOTICE).
