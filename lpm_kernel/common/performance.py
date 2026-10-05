"""Per-run synthesis timing and usage; never persist prompts or credentials."""
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
import uuid

logger = logging.getLogger(__name__)
RUN_DIRECTORY_ENV = "SECONDME_PERFORMANCE_RUN_DIR"
_append_lock = threading.Lock()


def _append_event(event):
    directory = os.environ.get(RUN_DIRECTORY_ENV)
    if not directory:
        return
    try:
        path = Path(directory) / "events.jsonl"
        with _append_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        # Metrics must not turn an otherwise successful synthesis into a retry.
        logger.warning("Could not persist performance metrics")


def _field(value, name, default=None):
    return (
        value.get(name, default)
        if isinstance(value, dict)
        else getattr(value, name, default)
    )


def _token_count(value):
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


def observe_chat_client(client, stage):
    """Keep the SDK/client contract while observing non-streaming completions."""
    create = client.chat.completions.create

    @wraps(create)
    def observed(*args, **kwargs):
        if not os.environ.get(RUN_DIRECTORY_ENV) or kwargs.get("stream"):
            return create(*args, **kwargs)
        started = time.perf_counter()
        event = {
            "kind": "completion",
            "stage": stage,
            "model": kwargs.get("model"),
            "status": "failed",
        }
        try:
            result = create(*args, **kwargs)
            event["status"] = "completed"
            usage = _field(result, "usage")
            prompt_details = _field(usage, "prompt_tokens_details")
            completion_details = _field(usage, "completion_tokens_details")
            event.update(
                prompt_tokens=_token_count(_field(usage, "prompt_tokens")),
                completion_tokens=_token_count(_field(usage, "completion_tokens")),
                provider_cached_input_tokens=_token_count(
                    _field(prompt_details, "cached_tokens")
                ),
                reasoning_tokens=_token_count(
                    _field(completion_details, "reasoning_tokens")
                ),
                usage_reported=usage is not None,
            )
            return result
        except Exception as exc:
            event["error_type"] = type(exc).__name__
            status_code = getattr(exc, "status_code", None)
            if isinstance(status_code, int):
                event["http_status"] = status_code
            raise
        finally:
            event["seconds"] = round(time.perf_counter() - started, 6)
            _append_event(event)

    client.chat.completions.create = observed
    return client


def create_synthesis_client(stage, **kwargs):
    from openai import OpenAI

    return observe_chat_client(OpenAI(**kwargs), stage)


class PerformanceRun:
    """One workflow's timing report, including failed/cancelled stages."""

    def __init__(self, root=None, metadata=None):
        root = Path(root) if root is not None else Path.cwd() / "data" / "performance"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = root / f"{stamp}-{uuid.uuid4().hex[:8]}"
        self.metadata = metadata or {}
        self.status = None
        self.started = time.perf_counter()
        self._previous_directory = None

    def __enter__(self):
        self._previous_directory = os.environ.get(RUN_DIRECTORY_ENV)
        os.environ[RUN_DIRECTORY_ENV] = str(self.directory)
        logger.info("Performance report directory: %s", self.directory)
        return self

    @contextmanager
    def step(self, name):
        started = time.perf_counter()
        outcome = {"kind": "step", "stage": name, "status": "failed"}
        try:
            yield outcome
        except Exception as exc:
            outcome["error_type"] = type(exc).__name__
            raise
        finally:
            outcome["seconds"] = round(time.perf_counter() - started, 6)
            _append_event(outcome)
            logger.info(
                "Step timing: %s %.2fs (%s)",
                name,
                outcome["seconds"],
                outcome["status"],
            )

    def __exit__(self, exception_type, exception, traceback):
        try:
            self._save_summary(exception_type)
        finally:
            if self._previous_directory is None:
                os.environ.pop(RUN_DIRECTORY_ENV, None)
            else:
                os.environ[RUN_DIRECTORY_ENV] = self._previous_directory

    def _save_summary(self, exception_type):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            events = []
            path = self.directory / "events.jsonl"
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    events.append(json.loads(line))
            steps = [event for event in events if event["kind"] == "step"]
            calls = [event for event in events if event["kind"] == "completion"]
            groups = defaultdict(list)
            for call in calls:
                groups[call["stage"]].append(call)
            usage = {}
            for stage, rows in groups.items():
                latency = sorted(row["seconds"] for row in rows)
                usage[stage] = {
                    "completion_calls": len(rows),
                    "failed_calls": sum(row["status"] != "completed" for row in rows),
                    "cumulative_call_seconds": round(sum(latency), 6),
                    "p50_seconds": latency[math.ceil(len(latency) * 0.5) - 1],
                    "p95_seconds": latency[math.ceil(len(latency) * 0.95) - 1],
                    **{
                        key: sum(row.get(key, 0) for row in rows)
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "provider_cached_input_tokens",
                            "reasoning_tokens",
                        )
                    },
                    "calls_with_usage": sum(
                        row.get("usage_reported", False) for row in rows
                    ),
                }
            status = self.status or ("failed" if exception_type else "completed")
            if any(step["status"] == "failed" for step in steps):
                status = "failed"
            elif any(step["status"] == "suspended" for step in steps):
                status = "suspended"
            report = {
                "status": status,
                "seconds": round(time.perf_counter() - self.started, 6),
                "metadata": self.metadata,
                "steps": steps,
                "synthesis": usage,
            }
            (self.directory / "summary.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except (OSError, ValueError):
            logger.warning("Could not persist performance summary")
