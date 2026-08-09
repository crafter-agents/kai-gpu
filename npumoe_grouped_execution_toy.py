#!/usr/bin/env python3
"""Deterministic toy simulation of NPUMoE grouped expert execution."""

from __future__ import annotations

from dataclasses import dataclass

from npumoe_capacity_tier_toy import (
    CAPACITY_TIERS,
    SEED,
    assign_tier,
    synthetic_loads,
)


# Eight is a small literal group size that creates full and remainder groups
# for this toy while keeping the group accounting easy to inspect.
GROUP_SIZE = 8
UNGROUPED_ZERO_PADDING_PERCENTAGE = 28.75


@dataclass(frozen=True)
class GroupStats:
    real_tokens: int
    padding_tokens: int
    row_budget: int

    @property
    def zero_padding_percentage(self) -> float:
        return 100 * self.padding_tokens / self.row_budget


def form_groups() -> dict[int, list[GroupStats]]:
    """Group assignments by tier and account for each packed row budget."""
    assignments = [
        assign_tier(load) for layer in synthetic_loads() for load in layer
    ]
    groups_by_tier: dict[int, list[GroupStats]] = {}

    for capacity in CAPACITY_TIERS:
        tier_assignments = [
            assignment
            for assignment in assignments
            if assignment.capacity == capacity
        ]
        groups: list[GroupStats] = []

        # A remainder forms one smaller final group, so no assignment is dropped.
        for start in range(0, len(tier_assignments), GROUP_SIZE):
            group_assignments = tier_assignments[start : start + GROUP_SIZE]
            row_budget = len(group_assignments) * capacity
            real_tokens = sum(
                assignment.real_tokens for assignment in group_assignments
            )
            padding_tokens = row_budget - real_tokens
            group = GroupStats(real_tokens, padding_tokens, row_budget)

            # Computing this property explicitly exercises the per-group metric.
            _ = group.zero_padding_percentage
            groups.append(group)

        groups_by_tier[capacity] = groups

    return groups_by_tier


def main() -> None:
    groups_by_tier = form_groups()

    print("NPUMoE grouped expert execution toy")
    print(f"seed: {SEED}")
    print(f"group size G: {GROUP_SIZE}")
    print(
        "groups per tier: "
        + ", ".join(
            f"{capacity}={len(groups_by_tier[capacity])}"
            for capacity in CAPACITY_TIERS
        )
    )

    combined_real = 0
    combined_padding = 0
    combined_budget = 0

    for capacity in CAPACITY_TIERS:
        groups = groups_by_tier[capacity]
        real_tokens = sum(group.real_tokens for group in groups)
        padding_tokens = sum(group.padding_tokens for group in groups)
        row_budget = sum(group.row_budget for group in groups)
        percentage = 100 * padding_tokens / row_budget
        combined_real += real_tokens
        combined_padding += padding_tokens
        combined_budget += row_budget
        print(
            f"tier {capacity}: real={real_tokens}, padding={padding_tokens}, "
            f"row budget={row_budget}, zero-padding={percentage:.2f}%"
        )

    combined_percentage = 100 * combined_padding / combined_budget
    print(f"combined real tokens: {combined_real}")
    print(f"combined padding tokens: {combined_padding}")
    print(f"combined row budget: {combined_budget}")
    print(f"combined zero-padding percentage: {combined_percentage:.2f}%")

    displayed_combined_percentage = round(combined_percentage, 2)
    if displayed_combined_percentage > UNGROUPED_ZERO_PADDING_PERCENTAGE:
        direction = "increased"
    elif displayed_combined_percentage < UNGROUPED_ZERO_PADDING_PERCENTAGE:
        direction = "decreased"
    else:
        direction = "left unchanged"
    print(
        f"grouping {direction} zero-padding versus the ungrouped baseline: "
        f"{combined_percentage:.2f}% vs {UNGROUPED_ZERO_PADDING_PERCENTAGE:.2f}%"
    )


if __name__ == "__main__":
    main()
