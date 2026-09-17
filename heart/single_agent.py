"""Single-learner adapter construction."""

from __future__ import annotations

from collections.abc import Sequence


def make_single_agent(
    mode: str = "classic-v0",
    *,
    controlled_player: int = 0,
    opponents: str | Sequence[str] = "medium",
    pass_opponents: str | Sequence[str] | None = None,
    play_opponents: str | Sequence[str] | None = None,
    **rule_overrides: object,
):
    """Create a learner-versus-rules environment for one seat.

    A three-item opponent sequence is assigned to the non-controlled player IDs
    in ascending order. A single difficulty name is broadcast to all three, and
    `pass_opponents`/`play_opponents` override it for one phase.
    """

    if mode != "classic-v0":
        raise ValueError(f"unknown mode {mode!r}; available modes: ('classic-v0',)")
    from heart.classic_single_agent import make_classic_single_agent

    return make_classic_single_agent(
        controlled_player=controlled_player,
        pass_opponents=opponents if pass_opponents is None else pass_opponents,
        play_opponents=opponents if play_opponents is None else play_opponents,
        **rule_overrides,
    )
