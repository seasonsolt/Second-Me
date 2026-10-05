"""Native MLX SFT adapter for the Web training pipeline (mlx-lm 0.28.3)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
from functools import partial


LORA_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def split_records(records, seed=42):
    """Keep every rewrite/turn of a source in one deterministic split."""
    groups = {}
    parents = list(range(len(records)))
    anchors = {}

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, sample in enumerate(records):
        keys = []
        for field in ("source_id", "doc_id"):
            if sample.get(field) is not None:
                keys.append(f"{field}:{sample[field]}")
        context = sample.get("context")
        if context:
            serialized = json.dumps(context, sort_keys=True, ensure_ascii=False)
            keys.append("context:" + hashlib.sha256(serialized.encode()).hexdigest())
        if not keys:
            source = sample.get("sample_id") or sample.get("id")
            if source is None:
                source = hashlib.sha256(json.dumps(sample, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            keys.append(f"sample:{source}")
        for key in keys:
            if key in anchors:
                parents[root(index)] = root(anchors[key])
            else:
                anchors[key] = index
    for index, sample in enumerate(records):
        groups.setdefault(root(index), []).append(sample)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("MLX SFT requires at least two independent source samples for train/validation")
    random.Random(seed).shuffle(keys)
    n_valid = max(1, round(len(keys) * 0.1))
    valid_keys = set(keys[:n_valid])
    return (
        [sample for key in keys if key not in valid_keys for sample in groups[key]],
        [sample for key in keys if key in valid_keys for sample in groups[key]],
    )


def training_iterations(n_train, epochs, batch_size, accumulation):
    """MLX iteration consumes one microbatch; updates happen each accumulation steps."""
    if n_train < 1 or epochs <= 0 or batch_size < 1 or accumulation < 1:
        raise ValueError("Training size, epochs, batch size, and accumulation must be positive")
    microbatches = math.ceil(math.ceil(n_train / batch_size) * epochs)
    # MLX does not flush a partial accumulation at the end.
    return math.ceil(microbatches / accumulation) * accumulation


def encode_assistant_sample(messages, tokenizer, max_length):
    """Apply the native nonthinking template once; mask all prompt tokens.

    Returns None when the sample exceeds max_length so callers can skip it.
    """
    from lpm_kernel.L2.chat_data import format_chat_completion

    formatted = format_chat_completion(messages, tokenizer)
    prefix = tokenizer.encode(formatted["prompt"], add_special_tokens=False)
    tokens = tokenizer.encode(formatted["prompt"] + formatted["completion"], add_special_tokens=False)
    if tokens[:len(prefix)] != prefix:
        raise ValueError("Native chat prompt is not a token prefix of the training sample")
    if len(tokens) > max_length:
        return None
    if len(tokens) <= len(prefix):
        raise ValueError("Training sample contains no completion tokens")
    return tokens, len(prefix)


class TokenDataset:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


def prepare_dataset(records, tokenizer, user_name, is_cot, max_length, output_file):
    from lpm_kernel.L2.chat_data import create_chat_messages, split_assistant_messages

    rows = []
    skipped = 0
    with open(output_file, "w", encoding="utf-8") as output:
        for record in records:
            messages = create_chat_messages(record, user_name, is_cot)
            for sample in split_assistant_messages(messages):
                row = encode_assistant_sample(sample, tokenizer, max_length)
                if row is None:
                    skipped += 1
                    continue
                rows.append(row)
                output.write(json.dumps({"messages": sample}, ensure_ascii=False) + "\n")
    if skipped:
        print(f"Skipped {skipped}/{skipped + len(rows)} samples longer than max_seq_length={max_length} in {Path(output_file).name}", flush=True)
    if not rows:
        raise ValueError(f"No usable assistant completions within max_seq_length={max_length} in {output_file}")
    return TokenDataset(rows)


def plan_batches(dataset, batch_size, train=False, group_by_length=True):
    """Group similar training lengths, then randomize batches rather than examples."""
    import numpy as np

    if batch_size < 1 or not len(dataset):
        raise ValueError("A nonempty dataset and positive batch size are required")
    indices = list(range(len(dataset)))
    if train and group_by_length:
        indices.sort(key=lambda index: len(dataset[index][0]))
    elif train:
        np.random.shuffle(indices)
    batches = [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]
    if train and group_by_length:
        np.random.shuffle(batches)
    return batches


def iterate_complete_batches(dataset, batch_size, max_seq_length, train=False,
                             group_by_length=True, statistics=None):
    """Use MLX's trainer with remainder batches and assistant-only loss boundaries."""
    import mlx.core as mx
    import numpy as np

    while True:
        for indices in plan_batches(dataset, batch_size, train, group_by_length):
            rows = [dataset[index] for index in indices]
            width = min(max_seq_length, max(len(tokens) for tokens, _ in rows))
            batch = np.zeros((len(rows), width), dtype=np.int32)
            boundaries = []
            for index, (tokens, offset) in enumerate(rows):
                if len(tokens) > max_seq_length:
                    raise ValueError("Training samples must be validated before batching")
                batch[index, :len(tokens)] = tokens
                # default_loss uses inclusive target positions; never train on padding.
                boundaries.append((offset, len(tokens) - 1))
            if train and statistics is not None:
                input_tokens = sum(len(tokens) for tokens, _ in rows)
                effective_input_tokens = input_tokens - len(rows)
                model_input_tokens = sum(min(len(tokens), width - 1) for tokens, _ in rows)
                padded_model_input_tokens = len(rows) * (width - 1)
                statistics.append({
                    "examples": len(rows), "input_tokens": input_tokens,
                    "model_input_tokens": model_input_tokens,
                    "effective_input_tokens": effective_input_tokens,
                    "supervised_tokens": sum(len(tokens) - offset for tokens, offset in rows),
                    "padded_model_input_tokens": padded_model_input_tokens,
                    "padding_tokens": padded_model_input_tokens - model_input_tokens,
                })
            yield mx.array(batch), mx.array(boundaries)
        if not train:
            return


class WebTrainingCallback:
    def __init__(self, iterations, output, statistics=None):
        self.iterations = iterations
        self.output = output
        self.statistics = statistics

    def on_train_loss_report(self, info):
        if not math.isfinite(info["train_loss"]):
            raise FloatingPointError("MLX training loss is not finite")
        iteration = info["iteration"]
        info = dict(info)
        if self.statistics is not None:
            info.update(self.statistics[iteration - 1])
        if info.get("iterations_per_second", 0) > 0:
            info["training_step_seconds"] = 1 / info["iterations_per_second"]
        # Completion is emitted only after adapter weights have been saved.
        percent = min(99, int(100 * iteration / self.iterations))
        print(f"{percent}%|##########| {iteration}/{self.iterations}", flush=True)
        with open(self.output, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(info) + "\n")

    def on_val_loss_report(self, info):
        if not math.isfinite(info["val_loss"]):
            raise FloatingPointError("MLX validation loss is not finite")
        with open(self.output, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(info) + "\n")


def run(args):
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx_lm import load
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    if args.lr <= 0 or args.max_length < 2:
        raise ValueError("learning_rate must be positive and max_seq_length at least two")
    if args.validation_batches == 0 or args.validation_batches < -1:
        raise ValueError("validation_batches must be -1 (all) or a positive batch count")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Invalidate completion before preflight; retries must not fuse a stale adapter.
    marker = output / "training_complete.json"
    marker.unlink(missing_ok=True)
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": False})
    with open(args.data, encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("merged.json must contain a list of training samples")
    if args.is_cot:
        print("Warning: --is-cot is ignored; training uses non-thinking responses", flush=True)
    train_records, valid_records = split_records(records, args.seed)
    dataset_dir = output / "data"
    dataset_dir.mkdir(exist_ok=True)
    train_set = prepare_dataset(train_records, tokenizer, args.user_name, args.is_cot, args.max_length, dataset_dir / "train.jsonl")
    valid_set = prepare_dataset(valid_records, tokenizer, args.user_name, args.is_cot, args.max_length, dataset_dir / "valid.jsonl")
    iterations = training_iterations(len(train_set), args.epochs, args.batch_size, args.grad_accumulation)
    if args.max_steps is not None:
        if args.max_steps < 1 or args.max_steps % args.grad_accumulation:
            raise ValueError("max_steps must be positive and divisible by grad_accumulation")
        iterations = min(iterations, args.max_steps)
    # Variable sample lengths leave freed buffers that are rarely reused; without a
    # limit the MLX cache grows to most of unified memory (53 GB seen on 64 GB).
    # Cap at 10% of device memory so smaller Macs keep headroom.
    mx.set_cache_limit(min(4 * 1024**3, mx.metal.device_info()["memory_size"] // 10))
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    model.freeze()
    # PEFT uses alpha/r; MLX directly multiplies the LoRA update by scale.
    lora_parameters = {"rank": 8, "scale": 16 / 8, "dropout": 0.1, "keys": LORA_KEYS}
    linear_to_lora_layers(model, len(model.layers), lora_parameters)
    adapter_config = {
        "fine_tune_type": "lora", "num_layers": len(model.layers),
        "lora_parameters": lora_parameters, "model": args.model,
    }
    (output / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2), encoding="utf-8")
    metrics = output / "training_metrics.jsonl"
    metrics.write_text("", encoding="utf-8")
    training_args = TrainingArgs(
        batch_size=args.batch_size, iters=iterations, val_batches=args.validation_batches,
        steps_per_report=1, steps_per_eval=max(1, iterations),
        steps_per_save=max(1, iterations), adapter_file=output / "adapters.safetensors",
        max_seq_length=args.max_length, grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation,
    )
    print("Training backend: MLX; objective: SFT; DPO: not executed", flush=True)
    print(f"Train samples: {len(train_set)}; valid samples: {len(valid_set)}; microbatches: {iterations}; optimizer updates: {iterations // args.grad_accumulation}", flush=True)
    print("***** Running training *****", flush=True)
    statistics = []
    started_at = time.perf_counter()
    train(
        model=model, optimizer=optim.AdamW(learning_rate=args.lr), args=training_args,
        train_dataset=train_set, val_dataset=valid_set,
        iterate_batches=partial(iterate_complete_batches, group_by_length=args.group_by_length, statistics=statistics),
        training_callback=WebTrainingCallback(iterations, metrics, statistics),
    )
    adapter = output / "adapters.safetensors"
    if not adapter.exists() or not adapter.stat().st_size:
        raise RuntimeError("MLX training exited without saved adapters")
    marker.write_text(json.dumps({
        "backend": "mlx", "objective": "sft", "dpo_executed": False,
        "model": str(Path(args.model).resolve()), "iterations": iterations,
        "train_samples": len(train_set), "valid_samples": len(valid_set),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "training_wall_seconds": time.perf_counter() - started_at,
        "group_by_length": args.group_by_length, "validation_batches": args.validation_batches,
        "batch_size": args.batch_size, "gradient_checkpointing": args.grad_checkpoint,
        "processed_examples": sum(step["examples"] for step in statistics),
        "input_tokens": sum(step["input_tokens"] for step in statistics),
        "model_input_tokens": sum(step["model_input_tokens"] for step in statistics),
        "effective_input_tokens": sum(step["effective_input_tokens"] for step in statistics),
        "supervised_tokens": sum(step["supervised_tokens"] for step in statistics),
        "padding_tokens": sum(step["padding_tokens"] for step in statistics),
        "padded_model_input_tokens": sum(step["padded_model_input_tokens"] for step in statistics),
    }, indent=2), encoding="utf-8")
    print(f"100%|##########| {iterations}/{iterations}", flush=True)
    print("=== Training Ended ===", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default="resources/L2/data/merged.json")
    parser.add_argument("--user-name", default="user")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--group-by-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--is-cot", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
