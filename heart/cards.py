"""Card indexing and constants shared by the engine and renderers."""

from __future__ import annotations

from numbers import Integral

import jax.numpy as jnp

NUM_PLAYERS = 4
NUM_SUITS = 4
NUM_RANKS = 13
NUM_CARDS = NUM_SUITS * NUM_RANKS
HAND_SIZE = NUM_CARDS // NUM_PLAYERS
CARDS_PER_PLAYER = 13
TRICKS_PER_DEAL = 13

CLUBS = 0
DIAMONDS = 1
SPADES = 2
HEARTS = 3

TWO = 0
QUEEN = 10


def card_id(suit: int, rank: int) -> int:
    """Return the canonical card ID for zero-based suit and rank indices."""

    if isinstance(suit, bool) or not isinstance(suit, Integral):
        raise TypeError("suit must be an integer")
    if isinstance(rank, bool) or not isinstance(rank, Integral):
        raise TypeError("rank must be an integer")
    suit = int(suit)
    rank = int(rank)
    if not 0 <= suit < NUM_SUITS:
        raise ValueError(f"suit must be in [0, {NUM_SUITS}), got {suit}")
    if not 0 <= rank < NUM_RANKS:
        raise ValueError(f"rank must be in [0, {NUM_RANKS}), got {rank}")
    return suit * NUM_RANKS + rank


TWO_OF_CLUBS = card_id(CLUBS, TWO)
QUEEN_OF_SPADES = card_id(SPADES, QUEEN)

CARD_SUITS = jnp.repeat(jnp.arange(NUM_SUITS, dtype=jnp.int8), NUM_RANKS)
CARD_RANKS = jnp.tile(jnp.arange(NUM_RANKS, dtype=jnp.int8), NUM_SUITS)
SUIT_MASKS = (
    jnp.arange(NUM_CARDS)[None, :] // NUM_RANKS == jnp.arange(NUM_SUITS)[:, None]
)
HEART_MASK = SUIT_MASKS[HEARTS]
POINT_CARD_MASK = HEART_MASK.at[QUEEN_OF_SPADES].set(True)

SUIT_SYMBOLS = ("♣", "♦", "♠", "♥")
RANK_NAMES = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
CARD_NAMES = tuple(
    f"{RANK_NAMES[rank]}{SUIT_SYMBOLS[suit]}"
    for suit in range(NUM_SUITS)
    for rank in range(NUM_RANKS)
)


def card_name(card: int) -> str:
    """Return a compact human-readable card name."""

    if isinstance(card, bool) or not isinstance(card, Integral):
        raise TypeError("card must be an integer")
    card = int(card)
    if not 0 <= card < NUM_CARDS:
        raise ValueError(f"card must be in [0, {NUM_CARDS}), got {card}")
    return CARD_NAMES[card]
