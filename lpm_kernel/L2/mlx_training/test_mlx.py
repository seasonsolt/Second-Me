"""Fast MLX pipeline contracts; no model, network, or personal configuration."""
import ast
import logging
import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lpm_kernel.L2.mlx_training.backend import resolve_training_backend, normalize_training_language
from lpm_kernel.L2.mlx_training.train import split_records, training_iterations


def isolated_service_class():
    """Load orchestration methods without initializing unrelated application services."""
    source = Path(__file__).parents[2] / "api/domains/trainprocess/trainprocess_service.py"
    tree = ast.parse(source.read_text())
    service = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = {
        "Optional": object, "Dict": dict, "os": __import__("os"),
        "subprocess": subprocess, "signal": signal, "threading": threading,
        "time": time, "re": __import__("re"), "logger": logging.getLogger(__name__),
        "Status": SimpleNamespace(SUSPENDED="suspended"),
    }
    # Annotations aren't necessary for these subprocess behavior tests.
    for node in ast.walk(service):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
            for argument in (*node.args.args, *node.args.kwonlyargs):
                argument.annotation = None
    exec(compile(ast.Module(body=[service], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["TrainProcessService"]


class MLXTrainingContracts(unittest.TestCase):
    def test_mac_selects_mlx_and_missing_install_is_explicit(self):
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            with patch("importlib.util.find_spec", return_value=object()):
                self.assertEqual(resolve_training_backend({}), "mlx")
            with patch("importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "requires mlx-lm"):
                    resolve_training_backend({})

    def test_other_platforms_keep_pytorch(self):
        with patch("platform.system", return_value="Linux"):
            self.assertEqual(resolve_training_backend({"use_cuda": True}), "pytorch")
            self.assertEqual(resolve_training_backend({"use_cuda": False}), "pytorch")
            with self.assertRaisesRegex(RuntimeError, "native macOS"):
                resolve_training_backend({"training_backend": "mlx"})
        with self.assertRaises(ValueError):
            resolve_training_backend({"training_backend": "mps"})

    def test_chinese_training_uses_the_existing_chinese_templates(self):
        for value in ("中文", "中文/Chinese", "Chinese/zh", "zh-TW"):
            self.assertEqual(normalize_training_language(value), "Chinese")
        self.assertEqual(normalize_training_language(None), "English")
        self.assertEqual(normalize_training_language("英文/English"), "English")

    def test_split_is_deterministic_and_keeps_source_rewrites_together(self):
        records = [{"source_id": str(index // 2), "assistant": str(index)} for index in range(20)]
        train, valid = split_records(records)
        self.assertEqual((train, valid), split_records(records))
        self.assertEqual(len(train) + len(valid), len(records))
        self.assertFalse({r["source_id"] for r in train} & {r["source_id"] for r in valid})
        with self.assertRaises(ValueError):
            split_records(records[:2])

    def test_doc_and_context_links_cannot_cross_splits(self):
        records = [
            {"source_id": "rewrite-1", "doc_id": "note-1", "context": "same memory", "assistant": "a"},
            {"source_id": "rewrite-2", "doc_id": "note-1", "context": "other chunk", "assistant": "b"},
            {"source_id": "rewrite-3", "doc_id": "note-2", "context": "same memory", "assistant": "c"},
            {"source_id": "independent", "context": "independent memory", "assistant": "d"},
        ]
        train, valid = split_records(records)
        side = {sample["assistant"]: "train" for sample in train}
        side.update({sample["assistant"]: "valid" for sample in valid})
        self.assertEqual(side["a"], side["b"])
        self.assertEqual(side["a"], side["c"])
        self.assertNotEqual(side["a"], side["d"])

    def test_iterations_cover_remainders_and_flush_accumulation(self):
        self.assertEqual(training_iterations(5, 3, 2, 1), 9)
        self.assertEqual(training_iterations(5, 3, 2, 4), 12)
        self.assertEqual(training_iterations(1, 1, 8, 1), 1)
        with self.assertRaises(ValueError):
            training_iterations(3, 1, 0, 1)

    def test_mlx_batches_keep_remainders_and_mask_padding(self):
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("MLX is available only in the native Apple Silicon environment")
        from lpm_kernel.L2.mlx_training.train import TokenDataset, iterate_complete_batches
        dataset = TokenDataset([([1, 2, 3, 4], 2), ([1, 2, 3], 1), ([1, 2, 3, 4, 5], 2)])
        batches = list(iterate_complete_batches(dataset, 2, 8))
        self.assertEqual([batch.shape[0] for batch, _ in batches], [2, 1])
        self.assertEqual(batches[0][1].tolist(), [[2, 3], [1, 2]])
        # Trainer target positions are inclusive; padding cannot enter the loss.
        targets = mx.arange(1, batches[0][0].shape[1])
        lengths = batches[0][1]
        mask = (targets >= lengths[:, 0:1]) & (targets <= lengths[:, 1:])
        self.assertEqual(mask.tolist(), [[False, True, True], [True, True, False]])

    def test_new_parameters_persist_with_legacy_defaults(self):
        path = Path(__file__).parents[2] / "api/domains/trainprocess/training_params_manager.py"
        spec = importlib.util.spec_from_file_location("isolated_training_params", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manager = module.TrainingParamsManager
        with tempfile.TemporaryDirectory() as directory:
            manager._params_file_path = str(Path(directory) / "params.json")
            manager.update_training_params({
                "training_backend": "mlx", "resolved_training_backend": "mlx",
                "max_steps": 2, "batch_size": 2, "gradient_accumulation_steps": 4,
                "training_language": "中文", "dpo_executed": False,
                "group_by_length": False, "validation_batches": 2,
            })
            params = manager.get_latest_training_params()
            self.assertEqual(params["training_backend"], "mlx")
            self.assertEqual(params["max_steps"], 2)
            self.assertEqual(params["gradient_accumulation_steps"], 4)
            self.assertEqual(params["training_language"], "中文")
            self.assertFalse(params["dpo_executed"])
            self.assertFalse(params["group_by_length"])
            self.assertEqual(params["validation_batches"], 2)
            self.assertEqual(params["model_name"], "Qwen3-1.7B")
            effective = manager.prepare_training_params({"model_name": "Qwen3-4B-Instruct-2507"})
            self.assertEqual(effective["training_backend"], "mlx")
            self.assertEqual(effective["batch_size"], 2)
            reset = manager.prepare_training_params({"training_backend": None}, use_previous_params=False)
            self.assertEqual(reset["training_backend"], "auto")
            self.assertEqual(reset["batch_size"], 1)


    def test_monitor_never_marks_completion_from_a_log(self):
        cls = isolated_service_class()
        service = object.__new__(cls)
        service.is_stopped = False
        service._training_done = threading.Event()
        service._training_done.set()
        service.progress = SimpleNamespace(mark_step_status=lambda *args: self.fail("Log must not complete training"))
        updates = []
        service._update_progress = lambda *args: updates.append(args)
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text("***** Running training *****\n100%|##########| 2/2\n=== Training Ended ===\n")
            self.assertTrue(service._monitor_training_progress(log))
        self.assertEqual(updates[-1][2], 99)

    def test_cancel_terminates_only_the_tracked_process_group(self):
        cls = isolated_service_class()
        service = object.__new__(cls)
        service.current_step = None
        service._process_lock = threading.Lock()
        service._training_done = threading.Event()
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        tracked = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        service.process = tracked
        try:
            self.assertTrue(service.stop_process())
            self.assertIsNotNone(tracked.poll())
            self.assertIsNone(unrelated.poll())
            self.assertTrue(service._training_done.is_set())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)
            if tracked.poll() is None:
                tracked.kill()
                tracked.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
