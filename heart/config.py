"""Immutable mode configuration."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

MAX_QUEEN_OF_SPADES_PENALTY = (2**15 - 1) - 13


@dataclass(frozen=True)
class SingleDealRules:
    """Rules for the initial fixed-horizon competitive mode."""

    queen_of_spades_penalty: int = 5
    forbid_first_trick_points: bool = True
    require_hearts_broken: bool = True
    shooting_the_moon: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.queen_of_spades_penalty, bool) or not isinstance(
            self.queen_of_spades_penalty, Integral
        ):
            raise TypeError("queen_of_spades_penalty must be an integer")
        if self.queen_of_spades_penalty < 0:
            raise ValueError("queen_of_spades_penalty must be non-negative")
        if self.queen_of_spades_penalty > MAX_QUEEN_OF_SPADES_PENALTY:
            raise ValueError(
                f"queen_of_spades_penalty must be at most {MAX_QUEEN_OF_SPADES_PENALTY}"
            )


SINGLE = SingleDealRules()
