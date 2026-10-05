"""Shared data contracts at the HF/MLX boundary; no GPU or network needed."""
from lpm_kernel.L2.chat_data import create_chat_messages, format_chat_completion, split_assistant_messages
from lpm_kernel.L2.mlx_training.train import encode_assistant_sample


class MLXTokenizerShape:
    """MLX TokenizerWrapper exposes encode but is deliberately not callable."""
    name_or_path = "Qwen/Qwen3-1.7B"
    chat_template = "enable_thinking"
    eos_token = "<|im_end|>"

    def encode(self, value, add_special_tokens=False):
        return list(value.encode("utf-8"))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "".join(f"{m['role']}:{m['content']}\n" for m in messages) + "assistant:\n<think>\n\n</think>\n\n"


def test_hf_and_mlx_use_identical_completion_boundary():
    tokenizer = MLXTokenizerShape()
    sample = {"user": "最新时间？", "assistant": "下午三点。", "references": [{"source": "日历", "content": "最新预约时间为下午三点。"}]}
    messages = create_chat_messages(sample, "Tester")
    row = format_chat_completion(messages, tokenizer)
    tokens, offset = encode_assistant_sample(messages, tokenizer, 2048)
    assert tokens == tokenizer.encode(row["prompt"] + row["completion"])
    assert offset == len(tokenizer.encode(row["prompt"]))
    assert "日历" in row["prompt"]
    assert "最新预约时间" in row["prompt"]
    assert row["completion"] == "下午三点。<|im_end|>"


def test_multiturn_mlx_targets_exclude_previous_assistant_from_loss():
    tokenizer = MLXTokenizerShape()
    messages = create_chat_messages({"messages": [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "user", "content": "第二问"},
        {"role": "assistant", "content": "第二答"},
    ]})
    samples = list(split_assistant_messages(messages))
    tokens, offset = encode_assistant_sample(samples[1], tokenizer, 2048)
    assert "第一答" in bytes(tokens[:offset]).decode()
    assert bytes(tokens[offset:]).decode() == "第二答<|im_end|>"
