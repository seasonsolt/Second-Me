"""Canonical training messages and native-template formatting for both backends."""
from lpm_kernel.L2.training_prompt import MEMORY_PROMPT, CONTEXT_PROMPT, JUDGE_PROMPT

def chat_template_kwargs(tokenizer):
    """Use the model's native template, with hybrid Qwen3 thinking disabled."""
    name = getattr(tokenizer, "name_or_path", "").lower()
    template = getattr(tokenizer, "chat_template", "") or ""
    if "qwen3" in name and "2507" not in name and "enable_thinking" in template:
        return {"enable_thinking": False}
    return {}


def create_chat_messages(sample, user_name="user", is_cot=False):
    """Normalize one synthesis record without mutating it or applying a template."""
    import copy
    import re
    if sample.get("messages"):
        messages = copy.deepcopy(sample["messages"])
    elif sample.get("assistant") is not None:
        messages = [
            {"role": "system", "content": MEMORY_PROMPT.format(user_name=user_name)},
            {"role": "user", "content": sample["user"]},
            {"role": "assistant", "content": sample["assistant"]},
        ]
    elif sample.get("enhanced_request") is not None:
        messages = [
            {"role": "system", "content": CONTEXT_PROMPT.format(user_name=user_name)},
            {"role": "user", "content": f"{user_name}'s request is: " + sample["user_request"]},
            {"role": "assistant", "content": sample["enhanced_request"]},
        ]
    elif sample.get("user_feedback") is not None:
        messages = [
            {"role": "system", "content": JUDGE_PROMPT.format(user_name=user_name)},
            {"role": "user", "content": f"{user_name}'s request is: " + sample["user_request"] + "\nExpert's response is: " + sample["expert_response"]},
            {"role": "assistant", "content": sample["user_feedback"]},
        ]
    else:
        return []
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str):
            raise ValueError("Training messages require a supported role and string content")
        if message["role"] == "assistant":
            # Old synthesis stores reasoning plus <answer>; only train the answer.
            content = re.sub(r"<think>.*?</think>", "", message["content"], flags=re.S).strip()
            answer = re.search(r"<answer>(.*?)</answer>", content, flags=re.S)
            if answer:
                content = answer.group(1).strip()
            if not content or content == "None" or "<think>" in content:
                return []
            message["content"] = content
    from lpm_kernel.L2.memory_prompt import build_memory_system_prompt
    references = sample.get("references") or []
    context = sample.get("context")
    if isinstance(context, str) and context.strip():
        references = [*references, {"source": sample.get("source", "training context"), "content": context}]
    system = next((m for m in messages if m["role"] == "system"), None)
    if system is None:
        system = {"role": "system", "content": MEMORY_PROMPT.format(user_name=user_name)}
        messages.insert(0, system)
    system["content"] = build_memory_system_prompt(system["content"], references)
    return messages


def split_assistant_messages(messages):
    """Emit each target turn with its preceding history for completion-only loss."""
    for index, message in enumerate(messages):
        if message["role"] == "assistant" and index and messages[index - 1]["role"] == "user":
            yield messages[:index + 1]


def format_chat_completion(messages, tokenizer):
    """Return native generation prefix and one answer, checking the token boundary."""
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True,
        **chat_template_kwargs(tokenizer),
    )
    completion = messages[-1]["content"].strip() + tokenizer.eos_token
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
    if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError("Native chat template has an invalid completion token boundary")
    return {"prompt": prompt, "completion": completion}


