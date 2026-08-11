#!/usr/bin/env python3
"""Deterministic toy simulation of NPUMoE training updates and freezes.

This synthetic accounting exercise does not reproduce Orion's real training
results, calibration data, or hardware measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

from npumoe_capacity_tier_toy import (
    NUM_EXPERTS,
    NUM_LAYERS,
    SEED,
    assign_tier,
    synthetic_loads,
)


# An expert updates only when its routed-token gradient mass reaches this
# fixed synthetic threshold during the training step.
UPDATE_THRESHOLD = 24


@dataclass(frozen=True)
class UpdateDecision:
    load: int
    retained_tokens: int
    action: str


def training_decisions() -> list[list[UpdateDecision]]:
    """Classify each expert-step load and account for retained-token work."""
    return [
        [
            UpdateDecision(
                load=load,
                retained_tokens=assign_tier(load).real_tokens,
                action="update" if load >= UPDATE_THRESHOLD else "freeze",
            )
            for load in step
        ]
        for step in synthetic_loads()
    ]


def main() -> None:
    decisions = training_decisions()
    flat_decisions = [decision for step in decisions for decision in step]
    updated = [decision for decision in flat_decisions if decision.action == "update"]
    frozen = [decision for decision in flat_decisions if decision.action == "freeze"]

    baseline_work = sum(decision.retained_tokens for decision in flat_decisions)
    skipped_work = sum(decision.retained_tokens for decision in frozen)
    performed_work = baseline_work - skipped_work

    print("NPUMoE training update-vs-freeze toy")
    print(f"seed: {SEED}")
    print(f"shape: {NUM_LAYERS} training steps x {NUM_EXPERTS} experts")
    print(f"update threshold: {UPDATE_THRESHOLD} routed tokens")

    for step_index, step in enumerate(decisions, start=1):
        for expert_index, decision in enumerate(step):
            print(
                f"step {step_index}, expert {expert_index}: "
                f"load={decision.load}, decision={decision.action}"
            )

    print(f"updated expert-steps: {len(updated)}")
    print(f"frozen expert-steps: {len(frozen)}")
    print(f"baseline retained-token update work: {baseline_work}")
    print(f"performed retained-token update work: {performed_work}")
    print(f"skipped retained-token update work: {skipped_work}")
    print(f"synthetic update compute saved: {100 * skipped_work / baseline_work:.2f}%")
    print(
        "This is synthetic accounting only; it does not reproduce Orion's "
        "real training results, calibration data, or hardware measurements."
    )


if __name__ == "__main__":
    main()
