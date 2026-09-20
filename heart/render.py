"""Host-side renderers kept outside the compiled training path."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from jax import device_get

from heart.cards import CARD_NAMES, NUM_PLAYERS
from heart.types import State


def _validate_viewer(viewer: int | None) -> int | None:
    if viewer is None:
        return None
    if isinstance(viewer, bool) or not isinstance(viewer, Integral):
        raise TypeError("viewer must be an integer or None")
    viewer = int(viewer)
    if not 0 <= viewer < NUM_PLAYERS:
        raise ValueError(f"viewer must be in [0, {NUM_PLAYERS}) or None")
    return viewer


def _terminal_rewards(state: State) -> np.ndarray:
    """Reconstruct terminal rewards from effective scores."""

    scores = np.asarray(state.scores, dtype=np.float32)
    return np.zeros_like(scores) - scores


def _visible_trick(state: State) -> tuple[np.ndarray, int, bool]:
    """Return cards, original leader, and whether this is the previous trick."""

    position = int(state.trick_position)
    if position > 0:
        return np.asarray(state.current_trick), int(state.leader), False

    completed = int(state.trick_index)
    if completed == 0:
        return np.asarray(state.current_trick), int(state.leader), False

    index = completed - 1
    cards = np.asarray(state.trick_history)[index]
    winner = int(np.asarray(state.trick_winners)[index])
    led_suit = int(cards[0]) // 13
    ranks = np.asarray(
        [int(card) % 13 if int(card) // 13 == led_suit else -1 for card in cards]
    )
    winner_offset = int(np.argmax(ranks))
    leader = (winner - winner_offset) % NUM_PLAYERS
    return cards, leader, True


def _cards_text(cards: np.ndarray) -> str:
    visible = [int(card) for card in cards if int(card) >= 0]
    return " ".join(CARD_NAMES[card] for card in visible) if visible else "—"


def render_ansi(state: State, viewer: int | None = 0) -> str:
    """Render one unbatched state as readable text from a seat's point of view.

    ``viewer`` names the local player, whose hand is the only one printed;
    the others show their card count. ``None`` reveals every hand.
    """

    viewer = _validate_viewer(viewer)
    state = device_get(state)

    hands = np.asarray(state.hands)
    visible_trick, _, previous = _visible_trick(state)
    penalties = np.asarray(state.penalties)
    scores = np.asarray(state.scores)
    lines = [
        (
            f"Trick {min(int(state.trick_index) + 1, 13)}/13 | "
            f"active=P{int(state.active_player)} | "
            f"hearts_broken={bool(state.hearts_broken)}"
        ),
        f"{'Last trick' if previous else 'Table'}: {_cards_text(visible_trick)}",
        "Penalties: "
        + "  ".join(f"P{i}={int(penalties[i])}" for i in range(NUM_PLAYERS)),
    ]
    for player in range(NUM_PLAYERS):
        cards = np.flatnonzero(hands[player])
        if viewer is None or player == viewer:
            contents = _cards_text(cards)
        else:
            contents = " ".join(["??"] * len(cards)) or "-"
        marker = (
            "→"
            if player == int(state.active_player) and not bool(state.terminated)
            else " "
        )
        identity = " (YOU)" if player == viewer else ""
        lines.append(f"{marker} P{player}{identity}: {contents}")
    if bool(state.terminated):
        rewards = _terminal_rewards(state)
        if int(state.moon_shooter) >= 0:
            lines.append(f"P{int(state.moon_shooter)} shot the moon")
        lines.append(
            "Effective scores: "
            + "  ".join(f"P{i}={int(scores[i])}" for i in range(NUM_PLAYERS))
        )
        lines.append(
            "Terminal rewards: "
            + "  ".join(f"P{i}={float(rewards[i]):+.1f}" for i in range(NUM_PLAYERS))
        )
        winners = np.flatnonzero(np.asarray(state.winner_mask))
        lines.append("Winner: " + ", ".join(f"P{int(player)}" for player in winners))
    return "\n".join(lines)
