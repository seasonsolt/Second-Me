"""CPU-only contracts for length buckets, validation ordering and token accounting."""

import numpy as np
import pytest

from lpm_kernel.L2.mlx_training.train import TokenDataset, plan_batches, WebTrainingCallback


@pytest.fixture(autouse=True)
def restore_numpy_random_state():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def test_length_buckets_keep_every_example_and_remainder():
    dataset = TokenDataset([(list(range(length)), 1) for length in [100, 10, 90, 20, 80, 30, 70]])
    np.random.seed(42)
    batches = plan_batches(dataset, 2, train=True)
    assert sorted(index for batch in batches for index in batch) == list(range(7))
    assert sorted(len(batch) for batch in batches) == [1, 2, 2, 2]
    lengths = sorted(tuple(len(dataset[index][0]) for index in batch) for batch in batches)
    assert lengths == [(10, 20), (30, 70), (80, 90), (100,)]


def test_limited_validation_is_not_biased_to_shortest_buckets():
    dataset = TokenDataset([(list(range(length)), 1) for length in [100, 10, 90, 20, 80]])
    assert plan_batches(dataset, 2, train=False, group_by_length=True) == [[0, 1], [2, 3], [4]]
    with pytest.raises(ValueError):
        plan_batches(dataset, 0, train=True)


def test_bucketing_reduces_padding_against_mixed_length_order():
    dataset = TokenDataset([(list(range(length)), 1) for length in [8, 256, 16, 240, 24, 224, 32, 208]])
    np.random.seed(42)
    grouped = plan_batches(dataset, 2, train=True, group_by_length=True)
    mixed = plan_batches(dataset, 2, train=False)

    def padded_tokens(batches):
        return sum(len(batch) * max(len(dataset[index][0]) for index in batch) for batch in batches)

    assert padded_tokens(grouped) < padded_tokens(mixed)
    assert padded_tokens(grouped) == 1056


def test_metrics_include_real_supervision_and_step_time(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    statistics = [{"examples": 2, "input_tokens": 100, "supervised_tokens": 15,
                   "padding_tokens": 12, "model_input_tokens": 98}]
    callback = WebTrainingCallback(1, metrics, statistics)
    callback.on_train_loss_report({"iteration": 1, "train_loss": 1.2, "iterations_per_second": 0.5})
    import json
    row = json.loads(metrics.read_text())
    assert row["supervised_tokens"] == 15
    assert row["input_tokens"] == 100
    assert row["training_step_seconds"] == 2
    with pytest.raises(FloatingPointError):
        callback.on_train_loss_report({"iteration": 1, "train_loss": float("nan")})
