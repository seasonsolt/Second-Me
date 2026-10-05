"""Concurrency settings and real thread scheduling with synthetic local replies."""
import importlib.util
import json
import logging
from pathlib import Path
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def generators(monkeypatch, tmp_path):
    def stub(name, **attributes):
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    stub("lpm_kernel.configs.logging", get_train_process_logger=lambda: logging.getLogger("synthetic-concurrency"))
    stub("lpm_kernel.api.services.user_llm_config_service", UserLLMConfigService=lambda: SimpleNamespace(get_available_llm=lambda: None))
    stub("lpm_kernel.common.performance", create_synthesis_client=lambda stage, **kwargs: None)
    helper = load(monkeypatch, "lpm_kernel.L2.data_pipeline.data_prep.synthesis_config",
                  "lpm_kernel/L2/data_pipeline/data_prep/synthesis_config.py")
    classes = []
    for name, cls in [("selfqa/selfqa_generator", "SelfQA"),
                      ("diversity/diversity_data_generator", "DiversityDataGenerator"),
                      ("preference/preference_QA_generate", "PreferenceQAGenerator")]:
        module = load(monkeypatch, "lpm_kernel.L2.data_pipeline.data_prep." + name.replace("/", "."),
                      "lpm_kernel/L2/data_pipeline/data_prep/" + name + ".py")
        classes.append(getattr(module, cls))
    topics = tmp_path / "topics.json"
    topics.write_text("{}")
    def construct():
        return [classes[0]("Synthetic User", "", "", is_cot=False),
                classes[1]("English", is_cot=False),
                classes[2](str(topics), "", "English", is_cot=False)]
    return helper, construct


@pytest.mark.parametrize("environment,expected", [({}, 2), ({"concurrency_threads": "3"}, 3),
                         ({"CONCURRENCY_THREADS": "4", "concurrency_threads": "3"}, 4)])
def test_constructors_use_shared_positive_integer(generators, monkeypatch, environment, expected):
    helper, construct = generators
    monkeypatch.delenv("CONCURRENCY_THREADS", raising=False)
    monkeypatch.delenv("concurrency_threads", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    assert helper.get_synthesis_workers() == expected
    assert [generator.max_workers for generator in construct()] == [expected] * 3


@pytest.mark.parametrize("invalid", ["0", "-1", "", "1.5", "many"])
def test_invalid_explicit_setting_fails_without_legacy_fallback(generators, monkeypatch, invalid):
    _, construct = generators
    monkeypatch.setenv("CONCURRENCY_THREADS", invalid)
    monkeypatch.setenv("concurrency_threads", "2")
    with pytest.raises(ValueError, match="positive integer"):
        construct()


class ActiveCalls:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def reply(self, *args, **kwargs):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        time.sleep(.02)
        with self.lock:
            self.active -= 1
        return "synthetic response"


@pytest.mark.parametrize("workers", [1, 3])
def test_three_generators_execute_at_configured_concurrency(generators, monkeypatch, tmp_path, workers):
    _, construct = generators
    monkeypatch.setenv("CONCURRENCY_THREADS", str(workers))
    selfqa, diversity, preference = construct()
    calls = ActiveCalls()
    selfqa._get_question_list = lambda: [f"synthetic question {index}" for index in range(9)]
    selfqa.get_remote_response = calls.reply
    assert len(selfqa.generate_qa()) == 9
    assert calls.peak == workers
    calls = ActiveCalls()
    def question(*args):
        calls.reply()
        return ["synthetic question"]
    def answer(*args):
        calls.reply()
        return "synthetic answer", "style"
    diversity._Q_generate = question
    diversity._A_generate = answer
    result = diversity._generate([{}] * 9, ["style"] * 9, None, {}, "English", "Synthetic User")
    assert len(result[0]) == 9 and len(result[1]) == 9
    assert calls.peak == workers
    calls = ActiveCalls()
    preference.data_synthesis_mode = "high"
    preference.pre_msg = {str(index): {"contents": ["topic: " + "synthetic content " * 6] * 20, "tags": []}
                          for index in range(9)}
    preference.generate_response = calls.reply
    output = tmp_path / "preference.json"
    preference.process_clusters(str(output))
    samples = json.loads(output.read_text())
    assert len(samples) == 18  # main plus large-cluster augmentation
    assert calls.calls == 36 and calls.peak == workers
    assert preference.question_list == samples


def test_preference_request_does_not_create_nested_thread_pool(generators, monkeypatch):
    _, construct = generators
    monkeypatch.setenv("CONCURRENCY_THREADS", "3")
    _, _, preference = construct()
    import concurrent.futures
    def fail_pool(*args, **kwargs):
        raise AssertionError("A single completion request must not open a thread pool")
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", fail_pool)
    preference.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="synthetic reply"))]))))
    assert preference.generate_response("synthetic rules", "synthetic input") == "synthetic reply"
