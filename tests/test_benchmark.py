from __future__ import annotations

import argparse

import jax
import pytest

from benchmarks.random_rollout import _positive_int, build_rollout
from benchmarks.single_agent_rollout import build_rollout as build_single_rollout


def test_benchmark_rejects_non_positive_sizes():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        build_rollout(0)
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        _positive_int("0")


@pytest.mark.parametrize(
    ("workload", "policy", "step_mode"),
    [
        ("engine-only", "random", "safe"),
        ("engine-only", "random", "trusted"),
        ("policy-inclusive", "random", "safe"),
    ],
)
def test_benchmark_workloads_return_scalar_checksum(workload, policy, step_mode):
    rollout = build_rollout(2, workload=workload, policy=policy, step_mode=step_mode)
    result = rollout(jax.random.key(0))
    jax.block_until_ready(result)
    assert result.shape == ()
    assert int(result) > 0


def test_benchmark_rejects_unknown_workload_and_policy():
    with pytest.raises(ValueError, match="unknown workload"):
        build_rollout(1, workload="mixed")
    with pytest.raises(ValueError, match="unknown policy"):
        build_rollout(1, policy="expert")
    with pytest.raises(ValueError, match="unknown step mode"):
        build_rollout(1, step_mode="unchecked-ish")


def test_single_agent_benchmark_returns_scalar_checksum():
    result = build_single_rollout(2, "easy")(jax.random.key(3))
    jax.block_until_ready(result)
    assert result.shape == ()
    assert int(result) > 0
