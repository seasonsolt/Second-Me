"""Local server ownership, build and prompt-budget contracts without live processes."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def local_service(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT))
    # Use the real DTO and service modules without eager importing all API routes.
    load_module(monkeypatch, "lpm_kernel.api.domains.kernel2.dto.server_dto",
                "lpm_kernel/api/domains/kernel2/dto/server_dto.py")
    from lpm_kernel.configs.config import Config
    settings = {
        "LOCAL_LLM_SERVICE_URL": "http://127.0.0.1:65401/v1",
        "LOCAL_LLM_CONTEXT_SIZE": "1024", "LOCAL_LLM_PARALLEL": "1",
    }
    monkeypatch.setattr(Config, "from_env", classmethod(
        lambda cls, *args, **kwargs: SimpleNamespace(get=lambda key, default=None: settings.get(key, default))))
    module = load_module(monkeypatch, "_secondme_local_llm_contracts",
                         "lpm_kernel/api/services/local_llm_service.py")
    monkeypatch.chdir(tmp_path)
    service = module.LocalLLMService()
    return module, service, settings


def fake_process(module, executable, port, pid):
    process = Mock()
    process.pid = pid
    process.alive = True
    process.cmdline.return_value = [str(executable), "--port", str(port), "-m", "/synthetic/model.gguf"]
    process.cwd.return_value = str(Path(executable).parent)
    process.oneshot.return_value = MagicMock()
    process.cpu_percent.return_value = 0
    process.memory_percent.return_value = 1
    process.create_time.return_value = 1
    process.terminate.side_effect = lambda: setattr(process, "alive", False)
    return process


def test_stop_manages_only_this_worktree_executable_and_port(local_service, monkeypatch):
    module, service, _ = local_service
    executable = service._server_path()
    own = fake_process(module, executable, 65401, 101)
    master = fake_process(module, Path("/synthetic/master/llama.cpp/build/bin/llama-server"), 65401, 102)
    other_port = fake_process(module, executable, 65402, 103)
    unrelated = fake_process(module, Path("/synthetic/other-app/llama-server"), 65401, 104)
    processes = [own, master, other_port, unrelated]
    monkeypatch.setattr(module.psutil, "process_iter", lambda _: [p for p in processes if p.alive])
    assert [p.pid for p in service._managed_processes()] == [101]
    assert not service.stop_server().is_running
    own.terminate.assert_called_once()
    for process in (master, other_port, unrelated):
        process.terminate.assert_not_called()
        process.kill.assert_not_called()


def install_fake_build(service, revision="current-revision", backend="metal"):
    model = Path.cwd() / "model.gguf"
    model.write_bytes(b"GGUF-synthetic")
    executable = service._server_path()
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"synthetic executable; never run")
    version = Path.cwd() / "dependencies/llama.cpp.version"
    version.parent.mkdir(parents=True)
    version.write_text("current-revision")
    marker = executable.parent.parent / ".secondme-build"
    marker.write_text(f"{revision} {backend}")
    return model


def test_obsolete_build_never_launches(local_service, monkeypatch):
    module, service, _ = local_service
    model = install_fake_build(service, revision="obsolete-revision")
    monkeypatch.setattr(service, "get_server_status", module.ServerStatus.not_running)
    popen = Mock()
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    assert not service.start_server(str(model))
    popen.assert_not_called()


def test_busy_unmanaged_port_is_refused_without_launch_or_cleanup(local_service, monkeypatch):
    module, service, _ = local_service
    model = install_fake_build(service)
    monkeypatch.setattr(service, "get_server_status", module.ServerStatus.not_running)
    monkeypatch.setattr(service, "_ensure_port_available", Mock(side_effect=RuntimeError("Port is already in use")))
    popen = Mock()
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    client = Mock()
    monkeypatch.setattr(module.httpx, "Client", client)
    assert not service.start_server(str(model))
    popen.assert_not_called()
    client.assert_not_called()


def test_port_probe_reports_conflict_without_touching_processes(local_service, monkeypatch):
    module, service, _ = local_service
    probe = MagicMock()
    probe.__enter__.return_value.bind.side_effect = OSError("Address already in use")
    monkeypatch.setattr(module.socket, "socket", Mock(return_value=probe))
    with pytest.raises(RuntimeError, match="already in use"):
        service._ensure_port_available("127.0.0.1", 65401)
    probe.__enter__.return_value.bind.assert_called_once_with(("127.0.0.1", 65401))


@pytest.mark.parametrize("build,system,cuda,use_gpu,layers", [
    ("metal", "Darwin", False, True, "999"),
    ("metal", "Darwin", False, False, "0"),
    ("metal", "Linux", False, True, "0"),
    ("cuda", "Linux", True, True, "999"),
    ("cuda", "Linux", False, True, "0"),
    ("cuda", "Linux", True, False, "0"),
    ("vulkan", "Linux", False, True, "999"),
    ("vulkan", "Linux", False, False, "0"),
    ("cpu", "Linux", True, True, "0"),
])
def test_offload_requires_matching_build_platform_and_user_choice(
    local_service, monkeypatch, build, system, cuda, use_gpu, layers,
):
    module, service, _ = local_service
    model = install_fake_build(service, backend=build)
    monkeypatch.setattr(service, "get_server_status", module.ServerStatus.not_running)
    # The implementation may expose a preflight so no test touches a real socket.
    monkeypatch.setattr(service, "_ensure_port_available", lambda *args: None, raising=False)
    monkeypatch.setattr(module.platform, "system", lambda: system)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    child = Mock()
    child.poll.return_value = None
    popen = Mock(return_value=child)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    client = MagicMock()
    client.__enter__.return_value.get.return_value.status_code = 200
    monkeypatch.setattr(module.httpx, "Client", Mock(return_value=client))
    try:
        assert service.start_server(str(model), use_gpu=use_gpu)
        command = popen.call_args.args[0]
        assert command[command.index("--n-gpu-layers") + 1] == layers
        assert command[command.index("--port") + 1] == "65401"
        assert "--jinja" in command
        assert popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    finally:
        if service._server_log is not None:
            service._server_log.close()


def test_different_model_request_does_not_stop_existing_server(local_service, monkeypatch):
    module, service, _ = local_service
    requested = install_fake_build(service)
    old_model = Path.cwd() / "old-model.gguf"
    old_model.write_bytes(b"old synthetic model")
    old_child = Mock()
    old_child.poll.return_value = None
    service._process = old_child
    status = module.ServerStatus.running(module.ProcessInfo(
        pid=101, cpu_percent=0, memory_percent=0, create_time=0,
        cmdline=[str(service._server_path()), "-m", str(old_model), "--port", "65401"],
    ))
    monkeypatch.setattr(service, "get_server_status", lambda: status)
    popen = Mock()
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    assert not service.start_server(str(requested))
    old_child.terminate.assert_not_called()
    old_child.kill.assert_not_called()
    popen.assert_not_called()
    assert service._process is old_child


def test_budget_removes_history_then_references_without_mutating_current_question(local_service, monkeypatch):
    module, service, _ = local_service
    messages = [
        {"role": "system", "content": "Identity rules.\n<reference_memories>\n"
         '<memory source="synthetic">' + "R" * 1000 + "</memory>\n</reference_memories>"},
        {"role": "user", "content": "Old question " + "X" * 500},
        {"role": "assistant", "content": "Old answer " + "Y" * 500},
        {"role": "user", "content": "Keep this entire current question."},
    ]
    original = copy.deepcopy(messages)
    calls = []

    def count(client, origin, fitted):
        calls.append(copy.deepcopy(fitted))
        return sum(len(message["content"]) for message in fitted)

    monkeypatch.setattr(service, "_count_prompt_tokens", count)
    monkeypatch.setattr(module.httpx, "Client", MagicMock())
    fitted, output_tokens = service.prepare_chat_request(messages, 2048)
    assert messages == original
    assert fitted[-1] == original[-1]
    assert [message["role"] for message in fitted] == ["system", "user"]
    assert fitted[0]["content"].startswith("Identity rules.")
    assert len(fitted[0]["content"]) < len(original[0]["content"])
    assert len(calls[1]) == 2  # History is removed before reference reduction.
    assert output_tokens == 256
    assert count(None, "", fitted) + output_tokens + 32 <= 1024


def test_slot_capacity_and_overlong_current_question(local_service, monkeypatch):
    module, service, settings = local_service
    settings["LOCAL_LLM_PARALLEL"] = "2"
    monkeypatch.setattr(module.httpx, "Client", MagicMock())
    monkeypatch.setattr(service, "_count_prompt_tokens", lambda client, origin, messages: sum(len(m["content"]) for m in messages))
    fitted, output_tokens = service.prepare_chat_request([{"role": "user", "content": "short question"}], 2048)
    assert output_tokens == 128  # Capacity is per slot, not total server context.
    assert fitted[0]["content"] == "short question"
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "Q" * 1000}]
    original = copy.deepcopy(messages)
    with pytest.raises(ValueError, match="shorten the question"):
        service.prepare_chat_request(messages, 2048)
    assert messages == original


def test_prompt_counter_uses_pinned_server_template_and_special_token_flags(local_service):
    _, service, _ = local_service
    apply_response = Mock()
    apply_response.json.return_value = {"prompt": "native-template-prompt"}
    token_response = Mock()
    token_response.json.return_value = {"tokens": [1, 2, 3]}
    client = Mock()
    client.post.side_effect = [apply_response, token_response]
    messages = [{"role": "user", "content": "synthetic"}]
    assert service._count_prompt_tokens(client, "http://127.0.0.1:65401", messages) == 3
    template_call, tokenize_call = client.post.call_args_list
    assert template_call.args[0].endswith("/apply-template")
    assert template_call.kwargs["json"]["messages"] == messages
    assert template_call.kwargs["json"]["add_generation_prompt"] is True
    assert template_call.kwargs["json"]["chat_template_kwargs"]["enable_thinking"] is False
    assert tokenize_call.kwargs["json"] == {
        "content": "native-template-prompt", "add_special": True, "parse_special": True,
    }
