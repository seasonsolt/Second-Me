"""Offline tests of real GraphRAG/fnllm file caches and semantic invalidation."""
import asyncio
from copy import deepcopy
import logging
from pathlib import Path
import subprocess

import httpx
from openai import AsyncOpenAI
import pytest
import yaml

from lpm_kernel.L2.data_pipeline.graphrag_indexing.runtime import (
    cache_namespace, prepare_runtime_settings, run_indexing_subprocess, stable_template_choice,
)
from lpm_kernel.L2.data_pipeline.graphrag_indexing.run import CountingCache, redact_secrets


def template():
    return {
        "models": {
            "default_chat_model": {"model": "synthetic-chat", "api_base": "https://offline.invalid/v1", "api_key": "fixture-not-a-key", "concurrent_requests": 2},
            "default_embedding_model": {"model": "synthetic-embed", "api_base": "https://offline.invalid/v1", "api_key": "fixture-not-a-key", "concurrent_requests": 2},
        },
        "input": {"base_dir": "input", "file_type": "text"},
        "output": {"base_dir": "output"}, "reporting": {"base_dir": "report"},
        "cache": {"type": "file", "base_dir": "cache"},
        "chunks": {"size": 500},
        "extract_graph": {"prompt": "prompts/extract_graph.txt"},
    }


def test_identical_input_has_stable_wrapper():
    choices = ["first {}", "second {}", "third {}"]
    original = stable_template_choice(choices, content="synthetic text", insight="synthetic insight", title="fixture")
    assert all(stable_template_choice(choices, content="synthetic text", insight="synthetic insight", title="fixture") == original for _ in range(20))


def test_namespace_changes_only_with_semantic_configuration():
    settings = template()
    prompts = {"prompts/extract_graph.txt": "Extract synthetic entities"}
    versions = {"graphrag": "fixture", "fnllm": "fixture"}
    base = cache_namespace(settings, prompts, package_versions=versions)
    operational = deepcopy(settings)
    operational["models"]["default_chat_model"].update(api_key="different-fixture", concurrent_requests=8)
    operational["output"]["base_dir"] = "different-output"
    operational["input"]["base_dir"] = "different-input"
    assert cache_namespace(operational, prompts, package_versions=versions) == base
    for field in ("model", "api_base"):
        changed = deepcopy(settings)
        changed["models"]["default_chat_model"][field] = "different-" + field
        assert cache_namespace(changed, prompts, package_versions=versions) != base
    assert cache_namespace(settings, {"prompts/extract_graph.txt": "Changed prompt"}, package_versions=versions) != base


def test_runtime_config_is_credential_free_and_does_not_mutate_templates(tmp_path):
    prompt_dir = tmp_path / "lpm_kernel/L2/data_pipeline/graphrag_indexing/prompts"
    prompt_dir.mkdir(parents=True)
    source = prompt_dir / "extract_graph.txt"
    source.write_text("Extract in <lang>")
    settings = template()
    original = deepcopy(settings)
    models = {name: {"model": model["model"], "api_base": model["api_base"]} for name, model in settings["models"].items()}
    args = (tmp_path, settings, tmp_path / "input", tmp_path / "output")
    root, config, cache = prepare_runtime_settings(*args, "Chinese", models, 8)
    saved = yaml.safe_load(config.read_text())
    assert all(model["concurrent_requests"] == 8 for model in saved["models"].values())
    assert "fixture-not-a-key" not in config.read_text()
    assert all(model["api_key"].startswith("${GRAPHRAG_") for model in saved["models"].values())
    assert Path(saved["cache"]["base_dir"]).is_absolute()
    assert (root / "prompts/extract_graph.txt").read_text() == "Extract in Chinese"
    assert settings == original
    assert source.read_text() == "Extract in <lang>"
    _, _, cache_again = prepare_runtime_settings(*args, "Chinese", models, 3)
    assert cache_again == cache
    _, _, cache_changed = prepare_runtime_settings(*args, "English", models, 3)
    assert cache_changed != cache
    assert source.read_text() == "Extract in <lang>"


def test_real_fnllm_cache_reuses_reopened_storage_and_invalidates_input(tmp_path):
    from fnllm.openai import PublicOpenAIConfig, create_openai_chat_llm, create_openai_embeddings_llm
    from graphrag.cache.factory import CacheFactory
    from graphrag.index.llm.load_llm import GraphRagLLMCache

    calls = []
    def respond(request):
        # httpx MockTransport catches every request; no local socket or cloud access.
        calls.append(request.url.path)
        if request.url.path.endswith("/embeddings"):
            body = {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.2, 0.7]}], "model": "synthetic-embed", "usage": {"prompt_tokens": 1, "total_tokens": 1}}
        else:
            body = {"id": "fixture", "object": "chat.completion", "created": 1, "model": "synthetic-chat", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "synthetic answer"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        return httpx.Response(200, json=body)

    def reopen():
        return CountingCache(CacheFactory.create_cache("file", str(tmp_path), {"base_dir": "persistent"}))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = AsyncOpenAI(api_key="fixture-not-a-key", base_url="https://offline.invalid/v1", http_client=http)
            config = PublicOpenAIConfig(model="synthetic-chat", encoding="cl100k_base", max_retries=0)
            first = reopen()
            chat = create_openai_chat_llm(config, client=client, cache=GraphRagLLMCache(first).child("chat"))
            await chat("Synthetic note A")
            assert first.counters["chat_misses"] == 1
            reopened = reopen()
            chat = create_openai_chat_llm(config, client=client, cache=GraphRagLLMCache(reopened).child("chat"))
            await chat("Synthetic note A")
            assert reopened.counters["chat_hits"] == 1
            assert len(calls) == 1
            await chat("Synthetic note B")
            assert reopened.counters["chat_misses"] == 1
            embed_config = PublicOpenAIConfig(model="synthetic-embed", encoding="cl100k_base", max_retries=0)
            embeddings = create_openai_embeddings_llm(embed_config, client=client, cache=GraphRagLLMCache(reopened).child("embed"))
            await embeddings(["Synthetic note A"])
            await embeddings(["Synthetic note A"])
            assert reopened.counters["embedding_misses"] == 1
            assert reopened.counters["embedding_hits"] == 1
            assert len(calls) == 3

    asyncio.run(scenario())


def test_indexing_failure_logs_returncode_and_output_tails(monkeypatch, caplog):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(3, command, output="o" * 5000 + "STDOUT_END", stderr="ValueError: bad synthetic config")
    monkeypatch.setattr(subprocess, "run", fail)
    logger = logging.getLogger("graphrag-test")
    with caplog.at_level(logging.INFO, logger="graphrag-test"), pytest.raises(subprocess.CalledProcessError):
        run_indexing_subprocess(["bash", "fixture.sh"], cwd=".", env={}, logger=logger)
    message = caplog.records[-1].getMessage()
    assert "code 3" in message and "STDOUT_END" in message and "bad synthetic config" in message
    assert "o" * 4100 not in message


def test_indexing_success_logs_stderr_as_warning(monkeypatch, caplog):
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "summary", "deprecation note"))
    logger = logging.getLogger("graphrag-test")
    with caplog.at_level(logging.INFO, logger="graphrag-test"):
        run_indexing_subprocess(["bash", "fixture.sh"], cwd=".", env={}, logger=logger)
    assert [(r.levelname, "deprecation note" in r.getMessage()) for r in caplog.records] == [("INFO", False), ("WARNING", True)]


def test_failure_message_redacts_graphrag_keys():
    environ = {"GRAPHRAG_CHAT_API_KEY": "fixture-secret", "GRAPHRAG_EMBEDDING_API_KEY": "", "OTHER": "visible"}
    assert redact_secrets("auth fixture-secret failed visible", environ) == "auth *** failed visible"
