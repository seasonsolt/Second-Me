"""Thread-safe workflow metrics without prompts, credentials or live requests."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from types import SimpleNamespace

import pytest

from lpm_kernel.common.performance import (
    PerformanceRun,
    RUN_DIRECTORY_ENV,
    observe_chat_client,
)


def client_with(create):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_concurrent_usage_is_complete_and_request_contents_never_persist(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(RUN_DIRECTORY_ENV, raising=False)
    usage = SimpleNamespace(
        prompt_tokens=20,
        completion_tokens=4,
        prompt_tokens_details=SimpleNamespace(cached_tokens=10),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    response = SimpleNamespace(usage=usage, choices=[{"content": "PRIVATE-OUTPUT"}])
    client = observe_chat_client(client_with(lambda **kwargs: response), "preference")
    with PerformanceRun(tmp_path, {"model_name": "synthetic"}) as run:
        with run.step("preference") as step:

            def call(_):
                return client.chat.completions.create(
                    model="synthetic",
                    messages=[{"content": "PRIVATE-PROMPT"}],
                    api_key="PRIVATE-KEY",
                )

            with ThreadPoolExecutor(max_workers=4) as pool:
                assert all(result is response for result in pool.map(call, range(12)))
            step["status"] = "completed"
    assert RUN_DIRECTORY_ENV not in os.environ
    report = json.loads((run.directory / "summary.json").read_text())
    totals = report["synthesis"]["preference"]
    assert report["status"] == "completed"
    assert totals["completion_calls"] == 12
    assert totals["prompt_tokens"] == 240
    assert totals["completion_tokens"] == 48
    assert totals["provider_cached_input_tokens"] == 120
    assert totals["failed_calls"] == 0
    serialized = "".join(path.read_text() for path in run.directory.iterdir())
    assert all(
        value not in serialized
        for value in ("PRIVATE-PROMPT", "PRIVATE-OUTPUT", "PRIVATE-KEY")
    )


def test_failure_keeps_original_exception_and_records_only_safe_error_type(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(RUN_DIRECTORY_ENV, "previous-run")
    error = RuntimeError("PRIVATE-KEY and PRIVATE-PROMPT")

    def fail(**kwargs):
        raise error

    client = observe_chat_client(client_with(fail), "diversity")
    with pytest.raises(RuntimeError) as caught:
        with PerformanceRun(tmp_path) as run:
            with run.step("diversity"):
                client.chat.completions.create(model="synthetic")
    assert caught.value is error
    assert os.environ[RUN_DIRECTORY_ENV] == "previous-run"
    report = json.loads((run.directory / "summary.json").read_text())
    assert report["status"] == "failed"
    assert report["synthesis"]["diversity"]["failed_calls"] == 1
    events = (run.directory / "events.jsonl").read_text()
    assert "RuntimeError" in events and "PRIVATE-" not in events


def test_no_run_and_streaming_calls_are_passthrough(tmp_path, monkeypatch):
    monkeypatch.delenv(RUN_DIRECTORY_ENV, raising=False)
    result = object()
    client = observe_chat_client(client_with(lambda **kwargs: result), "selfqa")
    assert client.chat.completions.create(model="synthetic") is result
    with PerformanceRun(tmp_path) as run:
        assert client.chat.completions.create(model="synthetic", stream=True) is result
        run.status = "suspended"
    report = json.loads((run.directory / "summary.json").read_text())
    assert report["status"] == "suspended" and report["synthesis"] == {}


def test_partial_stage_return_failure_is_reported_without_an_exception(
    tmp_path, monkeypatch
):
    monkeypatch.delenv(RUN_DIRECTORY_ENV, raising=False)
    with PerformanceRun(tmp_path) as run:
        with run.step("train") as step:
            step["status"] = "failed"
    report = json.loads((run.directory / "summary.json").read_text())
    assert report["status"] == "failed"
    assert report["steps"][0]["stage"] == "train"


def test_metric_io_failure_does_not_repeat_a_successful_completion(
    tmp_path, monkeypatch
):
    occupied = tmp_path / "not-a-directory"
    occupied.write_text("file")
    monkeypatch.setenv(RUN_DIRECTORY_ENV, str(occupied))
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage=None)

    client = observe_chat_client(client_with(create), "topics")
    assert client.chat.completions.create(model="synthetic").usage is None
    assert len(calls) == 1


@pytest.mark.parametrize("outcome", ["completed", "failed", "suspended"])
def test_workflow_applies_concurrency_before_graph_and_saves_terminal_status(
    tmp_path, monkeypatch, outcome
):
    from lpm_kernel.L2.data_pipeline.data_prep.synthesis_config import (
        get_synthesis_workers,
    )
    from lpm_kernel.L2.mlx_training.test_mlx import isolated_service_class

    monkeypatch.delenv(RUN_DIRECTORY_ENV, raising=False)
    monkeypatch.setenv("CONCURRENCY_THREADS", "2")
    monkeypatch.setenv("DATA_SYNTHESIS_MODE", "low")
    cls = isolated_service_class()
    params = {
        "model_name": "synthetic",
        "concurrency_threads": 6,
        "data_synthesis_mode": "medium",
    }
    updates = []
    stage = SimpleNamespace(
        value="map_your_entity_network",
        get_method_name=lambda: "map_your_entity_network",
    )
    namespace = cls.start_process.__globals__
    namespace.update(
        {
            "PerformanceRun": lambda metadata: PerformanceRun(tmp_path, metadata),
            "TrainingParamsManager": SimpleNamespace(
                get_latest_training_params=lambda: params,
                update_training_params=lambda value: updates.append(value),
            ),
            "resolve_training_backend": lambda value: "mlx",
            "get_synthesis_workers": get_synthesis_workers,
            "ProcessStep": SimpleNamespace(get_ordered_steps=lambda: [stage]),
            "Status": SimpleNamespace(FAILED="failed", SUSPENDED="suspended"),
        }
    )
    service = object.__new__(cls)
    marked = []
    service.progress = SimpleNamespace(
        get_last_successful_step=lambda: None,
        mark_step_status=lambda *args: marked.append(args),
    )

    def graph():
        assert os.environ["CONCURRENCY_THREADS"] == "6"
        assert os.environ["DATA_SYNTHESIS_MODE"] == "medium"
        service.is_stopped = outcome == "suspended"
        return outcome == "completed"

    service.map_your_entity_network = graph
    assert service.start_process() == (outcome == "completed")
    reports = list(tmp_path.glob("*/summary.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["status"] == outcome
    assert report["steps"][0]["status"] == outcome
    assert updates == [{"resolved_training_backend": "mlx"}]
    assert RUN_DIRECTORY_ENV not in os.environ
