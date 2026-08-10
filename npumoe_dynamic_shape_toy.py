#!/usr/bin/env python3
"""Compare exact dynamic shapes with NPUMoE static capacity tiers."""

from __future__ import annotations

from npumoe_capacity_tier_toy import (
    NUM_EXPERTS,
    NUM_LAYERS,
    SEED,
    assign_tier,
    synthetic_loads,
)


def main() -> None:
    loads = synthetic_loads()
    flat_loads = [load for layer in loads for load in layer]
    static_assignments = [assign_tier(load) for load in flat_loads]

    dynamic_real = sum(flat_loads)
    dynamic_padding = 0
    dynamic_overflow = 0
    dynamic_computed_slots = dynamic_real

    static_real = sum(
        assignment.real_tokens for assignment in static_assignments
    )
    static_padding = sum(
        assignment.padding_tokens for assignment in static_assignments
    )
    static_overflow = sum(
        assignment.overflow_tokens for assignment in static_assignments
    )
    static_computed_slots = static_real + static_padding
    static_zero_padding_percentage = (
        100 * static_padding / static_computed_slots
    )

    print("NPUMoE dynamic shape toy")
    print(f"seed: {SEED}")
    print(f"shape: {NUM_LAYERS} layers x {NUM_EXPERTS} experts")
    print("dynamic capacity: exact observed per-expert load")
    print(f"dynamic total real tokens: {dynamic_real}")
    print(f"dynamic total padding tokens: {dynamic_padding}")
    print(f"dynamic total overflow-pruned tokens: {dynamic_overflow}")
    print(f"dynamic total computed slots: {dynamic_computed_slots}")
    print("static-tier baseline:")
    print(f"static total real tokens: {static_real}")
    print(f"static total padding tokens: {static_padding}")
    print(f"static total overflow-pruned tokens: {static_overflow}")
    print(f"static total computed slots: {static_computed_slots}")
    print(
        "static zero-padding percentage: "
        f"{static_zero_padding_percentage:.2f}%"
    )


if __name__ == "__main__":
    main()
