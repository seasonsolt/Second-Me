"""Stable, credential-free GraphRAG settings and cache namespaces."""
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import yaml


# Operational limits do not change the semantic request or its cached result.
MODEL_OPERATIONAL_KEYS = {"api_key", "concurrent_requests", "requests_per_minute", "tokens_per_minute", "max_retries", "max_retry_wait", "request_timeout", "async_mode"}


def stable_template_choice(templates, *, content, insight=None, title=None):
    """Identical note content must not get a random GraphRAG input wrapper."""
    if not templates:
        raise ValueError("No note templates configured")
    value = json.dumps([content, insight, title], ensure_ascii=False, sort_keys=True)
    number = int(hashlib.sha256(value.encode()).hexdigest(), 16)
    return templates[number % len(templates)]


def cache_namespace(settings, prompts, *, package_versions=None):
    """Input text is keyed by fnllm; namespace covers provider and prompt semantics."""
    semantic = deepcopy(settings)
    for field in ("cache", "input", "output", "reporting", "update_index_output", "root_dir"):
        semantic.pop(field, None)
    semantic["models"] = {
        name: {key: value for key, value in model.items() if key not in MODEL_OPERATIONAL_KEYS}
        for name, model in settings["models"].items()
    }
    # Input parsing changes can affect chunk semantics even with identical documents.
    semantic["input"] = {key: value for key, value in settings.get("input", {}).items() if key != "base_dir"}
    versions = package_versions or {name: version(name) for name in ("graphrag", "fnllm")}
    data = {"schema": 1, "settings": semantic, "prompts": prompts, "versions": versions}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def prepare_runtime_settings(project_root, template, graph_input, output, language, models, workers):
    """Write placeholders, never credentials; retain cache independently of output."""
    if workers < 1:
        raise ValueError("GraphRAG concurrency must be positive")
    project_root = Path(project_root).resolve()
    graph_input = Path(graph_input).resolve()
    output = Path(output).resolve()
    settings = deepcopy(template)
    for name, config in settings["models"].items():
        config.update(models[name])
        config["model"] = config["model"].removeprefix("openai/")
        config["concurrent_requests"] = workers
        config["api_key"] = "${GRAPHRAG_CHAT_API_KEY}" if name == "default_chat_model" else "${GRAPHRAG_EMBEDDING_API_KEY}"
    prompt_root = project_root / "lpm_kernel/L2/data_pipeline/graphrag_indexing"
    prompts = {}
    for config in settings.values():
        if isinstance(config, dict) and config.get("prompt"):
            relative = config["prompt"]
            prompts[relative] = (prompt_root / relative).read_text(encoding="utf-8").replace("<lang>", language)
    namespace = cache_namespace(settings, prompts)
    # Use a stable source-directory identity, without putting note content in paths.
    source = hashlib.sha256(str(graph_input).encode()).hexdigest()[:12]
    runtime_root = project_root / "resources/L1/graphrag_runtime" / source
    cache_root = project_root / "resources/L1/graphrag_cache" / namespace
    runtime_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    settings["cache"] = {"type": "file", "base_dir": str(cache_root)}
    settings["input"]["base_dir"] = str(graph_input)
    settings["output"]["base_dir"] = str(output)
    settings["reporting"]["base_dir"] = str(output.parent / "report")
    for relative, text in prompts.items():
        path = runtime_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for config in settings.values():
        if isinstance(config, dict) and config.get("prompt"):
            config["prompt"] = str(runtime_root / config["prompt"])
    config_path = runtime_root / "settings.yaml"
    config_path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return runtime_root, config_path, cache_root


OUTPUT_TAIL_CHARS = 4000


def run_indexing_subprocess(command, *, cwd, env, logger):
    """Run the indexer, keeping the output tail in logs because the child disables GraphRAG logging."""
    try:
        result = subprocess.run(command, cwd=cwd, env=env, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        logger.error("GraphRAG indexing exited with code %s\nstdout tail:\n%s\nstderr tail:\n%s", error.returncode,
                     (error.stdout or "")[-OUTPUT_TAIL_CHARS:], (error.stderr or "")[-OUTPUT_TAIL_CHARS:])
        raise
    # The runner prints only counters/timings, never input documents or credentials.
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.warning("GraphRAG indexing stderr tail:\n%s", result.stderr[-OUTPUT_TAIL_CHARS:])
    return result
