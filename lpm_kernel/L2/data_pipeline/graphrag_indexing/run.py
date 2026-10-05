"""Run the pinned GraphRAG pipeline with real cache lookup counters, without .env."""
import argparse
import asyncio
from collections import Counter
import json
import logging
import os
from pathlib import Path
import sys
from string import Template
import threading
import time

import yaml
from graphrag.cache.pipeline_cache import PipelineCache


class CountingCache(PipelineCache):
    """Delegate every operation to GraphRAG's real persistent cache."""
    def __init__(self, cache, counters=None, lock=None):
        self.cache = cache
        self.counters = counters if counters is not None else Counter({"chat_hits": 0, "chat_misses": 0, "embedding_hits": 0, "embedding_misses": 0, "other_hits": 0, "other_misses": 0, "writes": 0})
        self.lock = lock if lock is not None else threading.Lock()

    async def get(self, key):
        value = await self.cache.get(key)
        family = "chat" if key.startswith("chat") else "embedding" if key.startswith("embeddings") else "other"
        with self.lock:
            self.counters[f"{family}_{'hits' if value is not None else 'misses'}"] += 1
        return value

    async def set(self, key, value, debug_data=None):
        await self.cache.set(key, value, debug_data)
        if value is not None:
            with self.lock:
                self.counters["writes"] += 1

    async def has(self, key):
        return await self.cache.has(key)

    async def delete(self, key):
        await self.cache.delete(key)

    async def clear(self):
        await self.cache.clear()

    def child(self, name):
        return CountingCache(self.cache.child(name), self.counters, self.lock)


async def run_pipeline_with_cache(config, cache):
    from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
    from graphrag.config.enums import IndexingMethod
    from graphrag.index.run.run_pipeline import run_pipeline
    from graphrag.index.workflows.factory import PipelineFactory

    pipeline = PipelineFactory.create_pipeline(config, IndexingMethod.Standard)
    failed = []
    async for result in run_pipeline(pipeline, config, cache=cache, callbacks=[NoopWorkflowCallbacks()]):
        if result.errors:
            failed.append(result.workflow)
    return failed


def redact_secrets(text, environ=None):
    environ = os.environ if environ is None else environ
    for name, value in environ.items():
        if name.startswith("GRAPHRAG_") and name.endswith("_API_KEY") and value:
            text = text.replace(value, "***")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    from graphrag.cache.factory import CacheFactory
    from graphrag.config.create_graphrag_config import create_graphrag_config

    # Suppress upstream prompt/config/error dumps; emit our own numeric metrics.
    logging.disable(logging.CRITICAL)
    # Resolve key placeholders after YAML parsing, without dotenv loading or serialization.
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for model in settings["models"].values():
        model["api_key"] = Template(model["api_key"]).substitute(os.environ)
    config = create_graphrag_config(settings, root_dir=str(args.root.resolve()))
    cache = CountingCache(CacheFactory.create_cache(config.cache.type, config.root_dir, config.cache.model_dump()))
    started = time.monotonic()
    failed = []
    success = False
    error_type = None
    try:
        failed = asyncio.run(run_pipeline_with_cache(config, cache))
        if failed:
            raise RuntimeError(f"GraphRAG workflow failed: {', '.join(failed)}")
        success = True
    except Exception as error:
        error_type = type(error).__name__
        raise
    finally:
        summary = {"elapsed_seconds": round(time.monotonic() - started, 3), "success": success,
                   "error_type": error_type, "failed_workflows": failed,
                   "cache_namespace": Path(config.cache.base_dir).name,
                   "concurrent_requests": config.models["default_chat_model"].concurrent_requests,
                   **dict(cache.counters)}
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("GraphRAG cache summary: " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Upstream logging is disabled, so this line is the only failure detail; keys are redacted.
        print(f"GraphRAG indexing failed ({type(error).__name__}): {redact_secrets(str(error))}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
