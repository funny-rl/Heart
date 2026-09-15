"""Immutable mode configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SingleDealRules:
    """Rules for the initial fixed-horizon competitive mode."""

    queen_of_spades_penalty: int = 5
    forbid_first_trick_points: bool = True
    require_hearts_broken: bool = True
    shooting_the_moon: bool = True

    def __post_init__(self) -> None:
        if self.queen_of_spades_penalty < 0:
            raise ValueError("queen_of_spades_penalty must be non-negative")


SINGLE = SingleDealRules()

