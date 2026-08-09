#!/usr/bin/env python3
"""Deterministic toy simulation of NPUMoE static capacity tiers."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass


SEED = 260418788
NUM_LAYERS = 6
NUM_EXPERTS = 12
CAPACITY_TIERS = (32, 64, 128)


@dataclass(frozen=True)
class Assignment:
    load: int
    capacity: int
    real_tokens: int
    padding_tokens: int
    overflow_tokens: int


def synthetic_loads() -> list[list[int]]:
    """Route 512 tokens per layer into reproducible, imbalanced loads."""
    rng = random.Random(SEED)

    expert_popularity = (
        0.28,
        0.40,
        0.55,
        0.72,
        0.90,
        1.10,
        1.35,
        1.65,
        2.00,
        2.45,
        3.00,
        3.70,
    )
    layer_factors = (0.82, 0.94, 1.06, 1.18, 1.31, 1.45)
    tokens_per_layer = 512
    noise_stddev = 2.5
    loads: list[list[int]] = []

    for layer_factor in layer_factors:
        expert_loads = [0] * NUM_EXPERTS
        for _ in range(tokens_per_layer):
            logits = [
                popularity * layer_factor + rng.gauss(0.0, noise_stddev)
                for popularity in expert_popularity
            ]
            max_logit = max(logits)
            exponentials = [math.exp(logit - max_logit) for logit in logits]
            total = sum(exponentials)
            probabilities = [value / total for value in exponentials]
            routed_expert = max(
                range(NUM_EXPERTS), key=probabilities.__getitem__
            )
            expert_loads[routed_expert] += 1
        loads.append(expert_loads)

    return loads


def assign_tier(load: int) -> Assignment:
    """Assign the smallest fitting tier, pruning above the largest tier."""
    capacity = next(
        (tier for tier in CAPACITY_TIERS if load <= tier),
        CAPACITY_TIERS[-1],
    )
    overflow_tokens = max(0, load - capacity)
    real_tokens = load - overflow_tokens
    return Assignment(
        load=load,
        capacity=capacity,
        real_tokens=real_tokens,
        padding_tokens=capacity - real_tokens,
        overflow_tokens=overflow_tokens,
    )


def main() -> None:
    loads = synthetic_loads()
    assignments = [assign_tier(load) for layer in loads for load in layer]
    tier_counts = Counter(assignment.capacity for assignment in assignments)

    total_real = sum(assignment.real_tokens for assignment in assignments)
    total_padding = sum(assignment.padding_tokens for assignment in assignments)
    total_overflow = sum(assignment.overflow_tokens for assignment in assignments)
    total_computed_slots = total_real + total_padding
    zero_padding_percentage = 100 * total_padding / total_computed_slots

    print("NPUMoE static capacity tier toy")
    print(f"seed: {SEED}")
    print(f"shape: {NUM_LAYERS} layers x {NUM_EXPERTS} experts")
    print(f"capacity tiers: {CAPACITY_TIERS}")
    print(
        "tier assignments: "
        + ", ".join(
            f"{tier}={tier_counts[tier]}" for tier in CAPACITY_TIERS
        )
    )
    print(f"total real tokens: {total_real}")
    print(f"total padding tokens: {total_padding}")
    print(f"total overflow-pruned tokens: {total_overflow}")
    print(f"total computed slots: {total_computed_slots}")
    print(f"zero-padding percentage: {zero_padding_percentage:.2f}%")


if __name__ == "__main__":
    main()
