#!/usr/bin/env python3
"""Deterministic toy simulation of NPUMoE compute graph residency ranking."""

from __future__ import annotations

from npumoe_grouped_execution_toy import (
    CAPACITY_TIERS,
    GROUP_SIZE,
    SEED,
    GroupStats,
    form_groups,
)


TaggedGroup = tuple[int, GroupStats]


def aggregate(groups: list[TaggedGroup]) -> tuple[int, int]:
    """Return real-token and row-budget totals for tagged groups."""
    return (
        sum(group.real_tokens for _, group in groups),
        sum(group.row_budget for _, group in groups),
    )


def main() -> None:
    groups_by_tier = form_groups()
    groups: list[TaggedGroup] = [
        (capacity, group)
        for capacity in CAPACITY_TIERS
        for group in groups_by_tier[capacity]
    ]

    # Python's sort is stable, so equal-demand groups retain the fixed order
    # established above by capacity tier and then by group within that tier.
    groups.sort(key=lambda tagged: tagged[1].real_tokens, reverse=True)

    # Keep exactly the top 1/2 resident, rounding up: ceil(len(groups) / 2).
    resident_count = -(-len(groups) // 2)
    resident = groups[:resident_count]
    fallback = groups[resident_count:]
    combined_real_tokens = sum(group.real_tokens for _, group in groups)

    resident_real, resident_budget = aggregate(resident)
    fallback_real, fallback_budget = aggregate(fallback)

    print(f"total groups: {len(groups)}")
    print(
        f"resident groups: {len(resident)}, fallback groups: {len(fallback)}"
    )
    print(
        f"resident: real tokens={resident_real}, row budget={resident_budget}, "
        f"combined-token coverage={100 * resident_real / combined_real_tokens:.2f}%"
    )
    print(
        f"fallback: real tokens={fallback_real}, row budget={fallback_budget}, "
        f"combined-token coverage={100 * fallback_real / combined_real_tokens:.2f}%"
    )
    print(
        "This models only demand ranking and placement classification; it does "
        "not reproduce the paper's real calibration traces, Figure 12 grouping "
        "choices, or Figure 17 measured 1.09x speedup from adding residency."
    )

    # These imported experiment constants define the reused deterministic input.
    _ = GROUP_SIZE, SEED


if __name__ == "__main__":
    main()
