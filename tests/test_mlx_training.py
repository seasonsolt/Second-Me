"""Collect MLX backend, parameters, progress and cancellation contracts with pytest."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from lpm_kernel.L2.mlx_training.test_mlx import MLXTrainingContracts as TestMLXTrainingContracts
from lpm_kernel.L2.mlx_training.train import prepare_dataset

# The shared unittest suite uses temporary parameter files and local patches;
# it does not import the application or its configured database/credentials.
__all__ = ["TestMLXTrainingContracts", "TestMLXDatasetLength", "TestTrainingParamValidation"]


class ByteTokenizer:
    """One token per UTF-8 byte keeps the prompt/completion boundary exact."""
    name_or_path = "byte-tokenizer"
    chat_template = ""
    eos_token = "</s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


class TestMLXDatasetLength(unittest.TestCase):
    def test_over_length_samples_are_skipped_not_fatal(self):
        tokenizer = ByteTokenizer()
        records = [
            {"user": "hi", "assistant": "short answer"},
            {"user": "hi", "assistant": "long answer " * 400},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "train.jsonl"
            dataset = prepare_dataset(records, tokenizer, "user", False, 2048, output)
            self.assertEqual(len(dataset), 1)
            self.assertTrue(all(len(tokens) <= 2048 for tokens, _ in dataset.rows))
            written = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0]["messages"][-1]["content"], "short answer")
            with self.assertRaisesRegex(ValueError, "No usable assistant completions"):
                prepare_dataset(records[1:], tokenizer, "user", False, 2048, output)


class TestTrainingParamValidation(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "lpm_kernel/api/domains/trainprocess/training_params_manager.py"
        spec = importlib.util.spec_from_file_location("isolated_training_params_validation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.manager = module.TrainingParamsManager

    def prepare(self, params):
        return self.manager.prepare_training_params(params, use_previous_params=False)

    def test_numeric_params_are_validated(self):
        self.assertEqual(self.prepare({"batch_size": 4, "max_steps": None})["batch_size"], 4)
        for params in (
            {"batch_size": 0}, {"batch_size": "2"}, {"batch_size": True},
            {"gradient_accumulation_steps": 65}, {"max_seq_length": 64},
            {"max_seq_length": 2048.5}, {"max_steps": 0},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                self.prepare(params)

    def test_thinking_mode_is_ignored(self):
        with self.assertLogs(level="WARNING"):
            self.assertFalse(self.prepare({"is_cot": True})["is_cot"])
