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
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

import heart
from heart.cards import NUM_CARDS, NUM_PLAYERS, QUEEN_OF_SPADES
from heart.classic import (
    MAX_CLASSIC_CORE_STEPS,
    NUM_PASS_ACTIONS,
    PASS,
    PASS_COMBINATIONS,
    TERMINAL,
    ClassicState,
    observe_classic,
)

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


def _classic_env():
    return heart.make("classic-v0")


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


def baseline_blob() -> bytes:
    """A floor to measure against: take the cheapest legal card, pass the top three.

    It reads only the contract's observation, so it is also the smallest
    worked example of a valid entry.
    """

    from jax import export

    hand = slice(*OFFSETS["hand"])
    legal = slice(*OFFSETS["legal"])
    rank_of = jnp.asarray([card % 13 for card in range(NUM_CARDS)], jnp.float32)

    def policy(observations):
        held = observations[:, hand]
        allowed = observations[:, legal]
        play = -rank_of[None, :] - 100.0 * (1.0 - allowed)
        # Passing indexes triples of held slots; prefer the highest cards.
        held_rank = held * (rank_of[None, :] + 1.0)
        order = jnp.argsort(-held_rank, axis=-1)[:, :13]
        slot_value = jnp.take_along_axis(held_rank, order, axis=-1)
        combinations = jnp.asarray(PASS_COMBINATIONS)
        triple = slot_value[:, combinations].sum(-1)
        return triple, play

    batch = export.symbolic_shape("b")[0]
    signature = jax.ShapeDtypeStruct((batch, OBSERVATION_DIM), jnp.float32)
    return export.export(jax.jit(policy))(signature).serialize()


@dataclass(frozen=True)
class Standing:
    """Where one entry finished, and how sure the league is about it."""

    rank: int
    name: str
    points: float
    stderr: float
    win_rate: float
    last_rate: float
    elo: float
    elo_stderr: float
    matches: int
    deals: int
    flops_per_decision: float


START_RATING = 1500.0
RATING_SCALE = 400.0


def _fit_elo(count, left, right, outcome, *, passes: int = 300) -> np.ndarray:
    """Ratings that reproduce the observed pairwise results.

    Hearts seats four, so each table is read as its six pairings. Sequential Elo
    would depend on the order games happened to be played, so the update repeats
    with a decaying step until it settles on the ratings the whole record
    implies.
    """

    ratings = np.full(count, START_RATING, np.float64)
    appearances = np.maximum(
        np.bincount(left, minlength=count) + np.bincount(right, minlength=count), 1
    )
    for index in range(passes):
        expected = 1.0 / (
            1.0 + 10.0 ** ((ratings[right] - ratings[left]) / RATING_SCALE)
        )
        residual = outcome - expected
        gradient = np.zeros(count, np.float64)
        np.add.at(gradient, left, residual)
        np.add.at(gradient, right, -residual)
        ratings += (32.0 / (1.0 + index / 25.0)) * gradient / appearances
        ratings -= ratings.mean() - START_RATING
    return ratings


def _pairings(seating: np.ndarray, scores: np.ndarray):
    """Every match becomes its six seat-versus-seat comparisons.

    Hearts is won by the lowest score, so the seat that finished the match with
    fewer penalty points takes the pairing.
    """

    left, right, outcome = [], [], []
    for first in range(NUM_PLAYERS):
        for second in range(first + 1, NUM_PLAYERS):
            left.append(seating[:, first])
            right.append(seating[:, second])
            gap = scores[:, second] - scores[:, first]
            outcome.append(np.where(gap > 0, 1.0, np.where(gap < 0, 0.0, 0.5)))
    return (
        np.concatenate(left),
        np.concatenate(right),
        np.concatenate(outcome).astype(np.float64),
    )


def _league_rollout(entries: list[Submission], batch: int, steps: int):
    """Play whole matches: a classic-v0 match runs to 100 points, which takes
    around ten deals, so a truncated scan would never reach its endgame."""

    env = _classic_env()
    reset = jax.vmap(env.reset)
    advance = jax.vmap(env.step)
    rows = jnp.arange(batch)

    @jax.jit
    def run(keys, seating):
        states, _ = reset(keys)

        def once(carry, _):
            state, deals = carry
            players = state.active_player.astype(jnp.int32)
            observations = jax.vmap(encode_observation)(state, players)
            passes = jnp.stack([entry.call(observations)[0] for entry in entries])
            plays = jnp.stack([entry.call(observations)[1] for entry in entries])
            acting = jnp.take_along_axis(seating, players[:, None], 1)[:, 0]
            match = jax.vmap(observe_classic)(state, players)
            pass_logits = jnp.where(
                match.pass_action_mask, passes[acting, rows], -jnp.inf
            )
            play_logits = jnp.where(
                match.play_action_mask, plays[acting, rows], -jnp.inf
            )
            actions = jnp.where(
                state.phase == PASS,
                jnp.argmax(pass_logits, -1),
                jnp.argmax(play_logits, -1),
            )
            following, _, _, _, info = advance(state, actions.astype(jnp.int32))
            return (following, deals + info.deal_completed), None

        (final, deals), _ = jax.lax.scan(
            once, (states, jnp.zeros((batch,), jnp.int32)), jnp.arange(steps)
        )
        # The league counts the game's own currency: penalty points, and the
        # match those points decide. A reward is this environment's
        # normalisation of them, so a leaderboard should not depend on it.
        return (
            final.match_scores.astype(jnp.float32),
            deals,
            final.winner_mask,
            final.phase == TERMINAL,
        )

    return run


def run_league(
    entries: list[Submission],
    *,
    lineups: int = 256,
    rounds: int = 4,
    steps: int = MAX_CLASSIC_CORE_STEPS,
    seed: int = 0,
    bootstrap: int = 16,
) -> list[Standing]:
    """Seat the entries against each other and rank them with their error.

    Every round reuses one set of deals for all seatings, and seats rotate
    within a lineup, so entries are compared on the same cards rather than on
    their luck. The reported figure is the mean normalized deal reward; two
    entries whose intervals overlap share a rank instead of being ordered by
    noise.
    """

    if len(entries) < NUM_PLAYERS:
        raise ValueError(f"a table needs {NUM_PLAYERS} entries")
    rollout = _league_rollout(entries, lineups, steps)
    rng = np.random.default_rng(seed)
    collected: dict[int, list[float]] = {index: [] for index in range(len(entries))}
    dealt: dict[int, int] = dict.fromkeys(range(len(entries)), 0)
    won: dict[int, int] = dict.fromkeys(range(len(entries)), 0)
    lost: dict[int, int] = dict.fromkeys(range(len(entries)), 0)
    pairings: list[tuple] = []

    for round_index in range(rounds):
        seating = np.stack(
            [rng.permutation(len(entries))[:NUM_PLAYERS] for _ in range(lineups)]
        )
        seating = np.roll(seating, round_index, axis=1)
        keys = jax.random.split(jax.random.key(seed + round_index), lineups)
        scores, deals, winners, over = (
            np.asarray(part)
            for part in jax.device_get(rollout(keys, jnp.asarray(seating)))
        )
        if not over.all():
            raise RuntimeError(
                f"{int((~over).sum())} matches did not finish in {steps} events"
            )
        per_deal = np.divide(scores, np.maximum(deals, 1)[:, None], dtype=np.float64)
        # Hearts is won by the lowest score, so the highest finishes last.
        losers = scores == scores.max(axis=1, keepdims=True)
        pairings.append(_pairings(seating, scores.astype(np.float64)))
        for seat in range(NUM_PLAYERS):
            for index in range(len(entries)):
                chosen = seating[:, seat] == index
                if chosen.any():
                    collected[index].extend(per_deal[chosen, seat])
                    won[index] += int(winners[chosen, seat].sum())
                    lost[index] += int(losers[chosen, seat].sum())
                    dealt[index] += int(deals[chosen].sum())

    left = np.concatenate([pair[0] for pair in pairings])
    right = np.concatenate([pair[1] for pair in pairings])
    outcome = np.concatenate([pair[2] for pair in pairings])
    ratings = _fit_elo(len(entries), left, right, outcome)

    generator = np.random.default_rng(seed + 1)
    resampled = [
        _fit_elo(
            len(entries),
            *(part[picks] for part in (left, right, outcome)),
            passes=120,
        )
        for picks in (
            generator.integers(0, left.size, left.size) for _ in range(bootstrap)
        )
    ]
    spread = (
        np.stack(resampled).std(axis=0, ddof=1)
        if len(resampled) > 1
        else np.zeros(len(entries))
    )

    standings = []
    for index, entry in enumerate(entries):
        values = np.asarray(collected[index], np.float64)
        mean = float(values.mean()) if values.size else 0.0
        error = (
            float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        )
        standings.append(
            Standing(
                0,
                entry.name,
                mean,
                error,
                won[index] / max(values.size, 1),
                lost[index] / max(values.size, 1),
                float(ratings[index]),
                float(spread[index]),
                int(values.size),
                dealt[index],
                entry.flops_per_decision,
            )
        )

    standings.sort(key=lambda s: s.points)
    ranked, rank = [], 0
    for position, standing in enumerate(standings):
        if position == 0:
            rank = 1
        else:
            previous = ranked[-1]
            gap = standing.points - previous.points
            spread = np.hypot(previous.stderr, standing.stderr)
            rank = previous.rank if gap <= spread else position + 1
        ranked.append(replace(standing, rank=rank))
    return ranked
