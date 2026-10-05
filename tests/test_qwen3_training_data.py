"""Exercise canonical data and the real local tokenizer/collator when downloaded."""
import copy
import json
from pathlib import Path

import pytest

from lpm_kernel.L2.chat_data import (
    create_chat_messages, format_chat_completion, split_assistant_messages,
)

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_messages_and_multiturn_are_not_mutated():
    samples = json.loads((ROOT / "tests/fixtures/qwen3_training.json").read_text())
    original = copy.deepcopy(samples)
    first = create_chat_messages(samples[0], "Tester")
    assert first[-1]["content"] == "先把最重要的一件事完成。"
    assert "<reference_memories>" in first[0]["content"]
    assert "本周工作日" in first[0]["content"]
    turns = list(split_assistant_messages(create_chat_messages(samples[1], "Tester")))
    assert [len(turn) for turn in turns] == [3, 5]
    assert samples == original


def test_empty_or_unclosed_reasoning_answers_are_excluded():
    assert not create_chat_messages({"user": "test", "assistant": "<think>unfinished"})
    assert not create_chat_messages({"user": "test", "assistant": "None"})


@pytest.mark.parametrize("model", ["Qwen3-1.7B", "Qwen3-4B-Instruct-2507"])
def test_native_template_and_completion_mask(model):
    model_path = ROOT / "resources/L2/base_models" / model
    if not (model_path / "tokenizer.json").is_file():
        pytest.skip("Local tokenizer has not been downloaded")
    from transformers import AutoTokenizer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    messages = create_chat_messages({"user": "怎么回复？", "assistant": "简洁回复。"})
    row = format_chat_completion(messages, tokenizer)
    prompt = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    full = tokenizer(row["prompt"] + row["completion"], add_special_tokens=False)["input_ids"]
    assert full[:len(prompt)] == prompt
    assert row["completion"].endswith(tokenizer.eos_token)
    if model == "Qwen3-1.7B":
        assert row["prompt"].endswith("<think>\n\n</think>\n\n")
    collator = DataCollatorForLanguageModeling(tokenizer.pad_token_id, completion_only_loss=True)
    batch = collator([{"input_ids": full, "completion_mask": [0] * len(prompt) + [1] * (len(full) - len(prompt))}])
    assert (batch["labels"][0, :len(prompt)] == -100).all()
    assert (batch["labels"][0, len(prompt):] != -100).all()
