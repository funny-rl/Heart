"""Host-side renderers kept outside the compiled training path."""

from __future__ import annotations

import numpy as np

from heart.cards import CARD_NAMES, NUM_PLAYERS
from heart.types import State


def _cards_text(cards: np.ndarray) -> str:
    visible = [int(card) for card in cards if int(card) >= 0]
    return " ".join(CARD_NAMES[card] for card in visible) if visible else "—"


def render_ansi(state: State, viewer: int | None = None) -> str:
    """Render one full-observability unbatched state as readable text.

    ``viewer`` identifies the local player but never hides any cards.
    """

    if viewer is not None and not 0 <= viewer < NUM_PLAYERS:
        raise ValueError(f"viewer must be in [0, {NUM_PLAYERS}) or None")

    hands = np.asarray(state.hands)
    current_trick = np.asarray(state.current_trick)
    penalties = np.asarray(state.penalties)
    scores = np.asarray(state.scores)
    lines = [
        (
            f"Trick {min(int(state.trick_index) + 1, 13)}/13 | "
            f"active=P{int(state.active_player)} | "
            f"hearts_broken={bool(state.hearts_broken)}"
        ),
        f"Table: {_cards_text(current_trick)}",
        "Scores: " + "  ".join(f"P{i}={int(penalties[i])}" for i in range(NUM_PLAYERS)),
    ]
    for player in range(NUM_PLAYERS):
        cards = np.flatnonzero(hands[player])
        contents = _cards_text(cards)
        marker = "→" if player == int(state.active_player) and not bool(state.terminated) else " "
        identity = " (YOU)" if player == viewer else ""
        lines.append(f"{marker} P{player}{identity}: {contents}")
    if bool(state.terminated):
        if int(state.moon_shooter) >= 0:
            lines.append(f"P{int(state.moon_shooter)} shot the moon")
            lines.append(
                "Effective scores: "
                + "  ".join(f"P{i}={int(scores[i])}" for i in range(NUM_PLAYERS))
            )
        winners = np.flatnonzero(np.asarray(state.winner_mask))
        lines.append("Winner: " + ", ".join(f"P{int(player)}" for player in winners))
    return "\n".join(lines)
