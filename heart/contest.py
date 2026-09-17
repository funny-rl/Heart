"""Submission contract for the leaderboard.

An entry is a policy exported to a portable computation graph rather than
Python. The host never imports submitter code: it deserialises the graph,
checks it against the declared signature, and runs it inside XLA. Host
callbacks cannot survive export, so a submission has no way out of the
computation.

The graph is free to be any architecture at all. Only the signature is fixed::

    (observations: float32[b, OBSERVATION_DIM])
        -> (pass_logits: float32[b, NUM_PASS_ACTIONS],
            play_logits: float32[b, NUM_CARDS])

with ``b`` exported as a symbolic dimension so the host chooses the batch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from heart.cards import NUM_CARDS, NUM_PLAYERS, QUEEN_OF_SPADES
from heart.classic import NUM_PASS_ACTIONS, PASS, ClassicState, observe_classic

# Observation layout, in order. Every segment is float32 and seat-relative:
# index 0 is always the acting player, 1 the seat to their left, and so on, so
# a policy never has to learn absolute seats.
SEGMENTS: tuple[tuple[str, int], ...] = (
    ("hand", NUM_CARDS),  # cards held
    ("played", NUM_CARDS),  # cards gone this deal
    ("table", NUM_CARDS),  # cards in the current trick
    ("legal", NUM_CARDS),  # cards the rules allow right now
    ("position", NUM_PLAYERS),  # how many have played before me
    ("taken_points", NUM_PLAYERS),  # deal penalties so far, /26
    ("match_scores", NUM_PLAYERS),  # cumulative scores, /100
    ("pass_direction", 4),  # left, right, across, hold
    ("phase", 2),  # passing, playing
    ("flags", 2),  # hearts broken, queen already played
)
OBSERVATION_DIM = sum(size for _, size in SEGMENTS)
OFFSETS = {}
_at = 0
for _name, _size in SEGMENTS:
    OFFSETS[_name] = (_at, _at + _size)
    _at += _size

MAX_SUBMISSION_BYTES = 8 * 1024 * 1024
MAX_FLOPS_PER_DECISION = 5_000_000
_CUSTOM_CALL = re.compile(r'call_target_name\s*=\s*"([^"]+)"')


class SubmissionError(ValueError):
    """A submission that the contract refuses to run."""


def encode_observation(state: ClassicState, player: int) -> jnp.ndarray:
    """Build the fixed-width view the contract promises a policy."""

    observation = observe_classic(state, player)
    game = state.game
    seats = (jnp.arange(NUM_PLAYERS) + player) % NUM_PLAYERS

    hand = game.hands[player].astype(jnp.float32)
    played = jnp.zeros(NUM_CARDS, jnp.float32)
    history = game.trick_history.reshape(-1)
    played = played.at[jnp.clip(history, 0, NUM_CARDS - 1)].set(
        jnp.where(history >= 0, 1.0, 0.0)
    )
    table = jnp.zeros(NUM_CARDS, jnp.float32)
    current = game.current_trick
    table = table.at[jnp.clip(current, 0, NUM_CARDS - 1)].set(
        jnp.where(current >= 0, 1.0, 0.0)
    )
    legal = observation.play_action_mask.astype(jnp.float32)

    position = jax.nn.one_hot(game.trick_position, NUM_PLAYERS, dtype=jnp.float32)
    taken = game.penalties[seats].astype(jnp.float32) / 26.0
    scores = state.match_scores[seats].astype(jnp.float32) / 100.0
    direction = jax.nn.one_hot(state.pass_direction, 4, dtype=jnp.float32)
    phase = jax.nn.one_hot(
        (state.phase != PASS).astype(jnp.int32), 2, dtype=jnp.float32
    )
    queen_gone = played[QUEEN_OF_SPADES]
    flags = jnp.stack([game.hearts_broken.astype(jnp.float32), queen_gone])
    return jnp.concatenate(
        [hand, played, table, legal, position, taken, scores, direction, phase, flags]
    )


@dataclass(frozen=True)
class Submission:
    """A validated entry, ready to be called on a batch of observations."""

    name: str
    call: object
    flops_per_decision: float
    size_bytes: int


def _cost(call, batch: int) -> float:
    sample = jnp.zeros((batch, OBSERVATION_DIM), jnp.float32)
    analysis = jax.jit(call).lower(sample).compile().cost_analysis()
    if isinstance(analysis, list):
        analysis = analysis[0] if analysis else {}
    return float(analysis.get("flops", 0.0)) / batch


def load_submission(
    blob: bytes,
    name: str,
    *,
    max_bytes: int = MAX_SUBMISSION_BYTES,
    max_flops: float = MAX_FLOPS_PER_DECISION,
    probe_batch: int = 64,
) -> Submission:
    """Validate an exported policy graph, or refuse it with a reason."""

    if not isinstance(blob, (bytes, bytearray)):
        raise SubmissionError("submission must be bytes")
    if len(blob) > max_bytes:
        raise SubmissionError(f"submission is {len(blob)} bytes, over {max_bytes}")

    try:
        from jax import export
    except ImportError as error:  # pragma: no cover - depends on the install
        raise SubmissionError("jax.export is unavailable") from error
    try:
        exported = export.deserialize(bytearray(blob))
    except Exception as error:
        raise SubmissionError(f"not a readable exported graph: {error}") from error

    if len(exported.in_avals) != 1:
        raise SubmissionError(
            f"expected one input, the observation batch; got {len(exported.in_avals)}"
        )
    shape = exported.in_avals[0].shape
    if len(shape) != 2 or str(shape[1]) != str(OBSERVATION_DIM):
        raise SubmissionError(
            f"input must be float32[b, {OBSERVATION_DIM}]; got {shape}"
        )
    if exported.in_avals[0].dtype != jnp.float32:
        raise SubmissionError(
            f"input must be float32; got {exported.in_avals[0].dtype}"
        )
    wanted = ((NUM_PASS_ACTIONS,), (NUM_CARDS,))
    got = tuple(tuple(str(d) for d in aval.shape[1:]) for aval in exported.out_avals)
    if got != tuple(tuple(str(d) for d in w) for w in wanted):
        raise SubmissionError(
            f"must return pass logits {wanted[0]} and play logits {wanted[1]}; got {got}"
        )
    for aval in exported.out_avals:
        if aval.dtype != jnp.float32:
            raise SubmissionError(f"outputs must be float32; got {aval.dtype}")

    targets = sorted(set(_CUSTOM_CALL.findall(exported.mlir_module())))
    if targets:
        raise SubmissionError(f"graph calls out of XLA: {targets}")

    flops = _cost(exported.call, probe_batch)
    if flops > max_flops:
        raise SubmissionError(
            f"{flops:,.0f} flops per decision is over the {max_flops:,.0f} budget"
        )

    probe = jax.jit(exported.call)(jnp.zeros((2, OBSERVATION_DIM), jnp.float32))
    if not all(bool(np.isfinite(np.asarray(part)).all()) for part in probe):
        raise SubmissionError("policy returned values that are not finite")

    return Submission(name, exported.call, flops, len(blob))
