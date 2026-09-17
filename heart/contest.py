"""Submission contract for the leaderboard.

An entry is a policy exported to a portable computation graph rather than
Python. The host never imports submitter code: it deserialises the graph,
checks it against the declared signature, and runs it inside XLA. Host
callbacks cannot survive export, so a submission has no way out of the
computation.

The graph may be any architecture. What is fixed is the interface, and it is
the environment's own — the observation a single learner receives and the
actions it may take::

    (observation: ClassicSingleAgentObservation)
        -> (pass_logits: float32[b, NUM_PASS_ACTIONS],
            play_logits: float32[b, HAND_SIZE])

with every leaf carrying a leading symbolic batch dimension, so the host
chooses the batch. Nothing here is a second encoding of the game: an entry sees
exactly what the packaged policies see, and a packaged policy is a valid entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

import heart
from heart.cards import NUM_CARDS, NUM_PLAYERS
from heart.classic import (
    MAX_CLASSIC_CORE_STEPS,
    NUM_PASS_ACTIONS,
    PASS,
    PLAY,
    TERMINAL,
    ClassicObservation,
    ClassicState,
)
from heart.classic_single_agent import ClassicSingleAgentObservation
from heart.single_agent import HAND_SIZE
from heart.types import Observation

MAX_SUBMISSION_BYTES = 8 * 1024 * 1024
MAX_FLOPS_PER_DECISION = 5_000_000
START_RATING = 1500.0
RATING_SCALE = 400.0
_CUSTOM_CALL = re.compile(r'call_target_name\s*=\s*"([^"]+)"')
_REGISTERED = False


class SubmissionError(ValueError):
    """A submission that the contract refuses to run."""


def _register() -> None:
    """Teach the exporter the observation's named tuples, once."""

    global _REGISTERED
    if _REGISTERED:
        return
    from jax import export

    for kind in (Observation, ClassicObservation, ClassicSingleAgentObservation):
        export.register_namedtuple_serialization(
            kind, serialized_name=f"heart.{kind.__name__}"
        )
    _REGISTERED = True


def sample_observation() -> ClassicSingleAgentObservation:
    """One unbatched observation, the shape every entry is exported against."""

    env = heart.make_classic_single_agent(
        controlled_player=0, pass_opponents="easy", play_opponents="easy"
    )
    return env.reset(jax.random.key(0))[1]


def observation_signature():
    """The batched input signature an entry is exported with."""

    from jax import export

    _register()
    batch = export.symbolic_shape("b")[0]
    return jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct((batch, *leaf.shape), leaf.dtype),
        sample_observation(),
    )


def host_platforms() -> tuple[str, ...]:
    """The platforms an entry should cover: always cpu, plus whatever is here.

    A device calls itself ``gpu`` while an export names the vendor, so the two
    have to be reconciled before they can be compared.
    """

    device = jax.devices()[0]
    platform = device.platform
    if platform == "gpu":
        platform = "rocm" if "rocm" in device.device_kind.lower() else "cuda"
    return tuple(dict.fromkeys(("cpu", platform)))


def export_policy(policy, *, platforms: tuple[str, ...] | None = None) -> bytes:
    """Export a policy written against the contract, ready to submit."""

    from jax import export

    _register()
    return export.export(jax.jit(policy), platforms=platforms or host_platforms())(
        observation_signature()
    ).serialize()


@dataclass(frozen=True)
class Submission:
    """A validated entry, ready to be called on a batch of observations."""

    name: str
    call: object
    flops_per_decision: float
    size_bytes: int


def _batched(observation, batch: int):
    return jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (batch, *leaf.shape)), observation
    )


def _cost(call, batch: int) -> float:
    sample = _batched(sample_observation(), batch)
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
    _register()
    try:
        exported = export.deserialize(bytearray(blob))
    except Exception as error:
        raise SubmissionError(f"not a readable exported graph: {error}") from error

    wanted = jax.tree.leaves(observation_signature())
    if len(exported.in_avals) != len(wanted):
        raise SubmissionError(
            f"expected the single-learner observation, {len(wanted)} arrays;"
            f" got {len(exported.in_avals)}"
        )
    for index, (given, expected) in enumerate(zip(exported.in_avals, wanted)):
        if given.dtype != expected.dtype or [str(d) for d in given.shape] != [
            str(d) for d in expected.shape
        ]:
            raise SubmissionError(
                f"observation array {index} must be {expected.dtype}{list(expected.shape)};"
                f" got {given.dtype}{list(given.shape)}"
            )

    outputs = ((NUM_PASS_ACTIONS,), (HAND_SIZE,))
    given = tuple(tuple(str(d) for d in aval.shape[1:]) for aval in exported.out_avals)
    if given != tuple(tuple(str(d) for d in shape) for shape in outputs):
        raise SubmissionError(
            f"must return pass logits {outputs[0]} over hand-slot triples and play"
            f" logits {outputs[1]} over hand slots; got {given}"
        )
    for aval in exported.out_avals:
        if aval.dtype != jnp.float32:
            raise SubmissionError(f"outputs must be float32; got {aval.dtype}")

    here = host_platforms()[-1]
    if here not in exported.platforms:
        raise SubmissionError(
            f"exported for {tuple(exported.platforms)} but this host runs on"
            f" '{here}'; export for {host_platforms()} so an entry runs wherever"
            " the league does"
        )

    targets = sorted(set(_CUSTOM_CALL.findall(exported.mlir_module())))
    if targets:
        raise SubmissionError(f"graph calls out of XLA: {targets}")

    flops = _cost(exported.call, probe_batch)
    if flops > max_flops:
        raise SubmissionError(
            f"{flops:,.0f} flops per decision is over the {max_flops:,.0f} budget"
        )

    probe = jax.jit(exported.call)(_batched(sample_observation(), 2))
    if not all(bool(np.isfinite(np.asarray(part)).all()) for part in probe):
        raise SubmissionError("policy returned values that are not finite")

    return Submission(name, exported.call, flops, len(blob))


def baseline_blob() -> bytes:
    """A floor to measure against: play the cheapest legal card, pass the top three.

    It reads only what the contract provides, so it doubles as the shortest
    worked example of an entry.
    """

    from heart.classic import PASS_COMBINATIONS

    combinations = jnp.asarray(PASS_COMBINATIONS)

    def policy(observation):
        cards = observation.play_hand_cards.astype(jnp.float32)
        held = observation.pass_hand_cards.astype(jnp.float32)
        play = -(cards % 13)
        triple = (held % 13)[:, combinations].sum(-1)
        return triple.astype(jnp.float32), play.astype(jnp.float32)

    return export_policy(policy)


@dataclass(frozen=True)
class Standing:
    """Where one entry finished, and how sure the league is about it."""

    rank: int
    name: str
    points: float
    stderr: float
    win_rate: float  # complete matches finished on the lowest cumulative score
    last_rate: float  # complete matches finished on the highest
    elo: float
    elo_stderr: float
    matches: int
    deals: int
    flops_per_decision: float


def _fit_elo(count, left, right, outcome, *, passes: int = 300) -> np.ndarray:
    """Ratings that reproduce the observed pairwise results.

    Hearts seats four, so each match is read as its six pairings. Sequential Elo
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


def _slots(state: ClassicState) -> jnp.ndarray:
    """Each seat's hand as thirteen slots, ascending, holes marked -1."""

    return jax.vmap(
        lambda held: jnp.nonzero(held, size=HAND_SIZE, fill_value=-1)[0].astype(
            jnp.int8
        )
    )(state.game.hands)


def _seat_observation(state: ClassicState, player, pass_hand, play_hand):
    """Rebuild what the single-learner adapter shows the seat about to act."""

    match = heart.classic.observe_classic(state, player)
    safe = jnp.clip(play_hand.astype(jnp.int32), 0, NUM_CARDS - 1)
    return ClassicSingleAgentObservation(
        match=match,
        pass_hand_cards=pass_hand,
        play_hand_cards=play_hand,
        pass_action_mask=match.pass_action_mask,
        play_action_mask=(play_hand >= 0) & match.play_action_mask[safe],
    )


def _league_step(entries: list[Submission], batch: int):
    """One event for a batch of matches, compiled on its own.

    The whole rollout was a single `lax.scan` until the fused program began
    offering actions the rules refuse — every part of it checks out alone and
    step by step, so the loop stays in Python and each step is compiled, which
    is slower and gives an answer that can be trusted.
    """

    env = heart.make("classic-v0")
    advance = jax.vmap(env.step)
    rows = jnp.arange(batch)

    @jax.jit
    def step(state, pass_hands, play_hands, seating):
        players = state.active_player.astype(jnp.int32)
        mine = jnp.take_along_axis(play_hands, players[:, None, None], 1)[:, 0]
        passed = jnp.take_along_axis(pass_hands, players[:, None, None], 1)[:, 0]
        observation = jax.vmap(_seat_observation)(state, players, passed, mine)

        passes = jnp.stack([entry.call(observation)[0] for entry in entries])
        plays = jnp.stack([entry.call(observation)[1] for entry in entries])
        acting = jnp.take_along_axis(seating, players[:, None], 1)[:, 0]
        pass_action = jnp.argmax(
            jnp.where(observation.pass_action_mask, passes[acting, rows], -jnp.inf), -1
        )
        slot = jnp.argmax(
            jnp.where(observation.play_action_mask, plays[acting, rows], -jnp.inf), -1
        )
        card = jnp.take_along_axis(mine, slot[:, None], 1)[:, 0].astype(jnp.int32)
        actions = jnp.where(state.phase == PASS, pass_action, card)
        following, _, _, _, info = advance(state, actions)

        # The adapter refreshes a layout when a deal opens, and again once the
        # passing phase has handed the cards over.
        fresh = jax.vmap(_slots)(following)
        new_deal = (following.deal_index != state.deal_index)[:, None, None]
        after_pass = ((state.phase == PASS) & (following.phase == PLAY))[
            :, None, None
        ] & ~new_deal
        return (
            following,
            jnp.where(new_deal, fresh, pass_hands),
            jnp.where(new_deal | after_pass, fresh, play_hands),
            info.deal_completed,
            info.invalid_action,
        )

    return step


def _play_matches(entries: list[Submission], keys, seating, steps: int):
    """Play a batch of matches to the end, refusing to guess if the rules do."""

    batch = seating.shape[0]
    env = heart.make("classic-v0")
    states, _ = jax.vmap(env.reset)(keys)
    hands = jax.vmap(_slots)(states)
    pass_hands = play_hands = hands
    step = _league_step(entries, batch)
    seating = jnp.asarray(seating)
    deals = np.zeros(batch, np.int64)
    refused = 0
    for _ in range(steps):
        # A finished match has no legal action left, so it is neither stepped
        # for its result nor counted when it declines to move.
        alive = np.asarray(jax.device_get(states.phase != TERMINAL))
        if not alive.any():
            break
        states, pass_hands, play_hands, settled, invalid = step(
            states, pass_hands, play_hands, seating
        )
        deals += np.asarray(jax.device_get(settled)) * alive
        refused += int((np.asarray(jax.device_get(invalid)) & alive).sum())
        if refused:
            break
    return (
        np.asarray(jax.device_get(states.match_scores), np.float64),
        deals,
        np.asarray(jax.device_get(states.winner_mask)),
        np.asarray(jax.device_get(states.phase == 2)),
        refused,
    )


def run_league(
    entries: list[Submission],
    *,
    lineups: int = 256,
    rounds: int = 4,
    steps: int = MAX_CLASSIC_CORE_STEPS,
    seed: int = 0,
    bootstrap: int = 16,
    chunk: int = 256,
) -> list[Standing]:
    """Seat the entries against each other and rank them with their error.

    Every round reuses one set of deals for all seatings, and seats rotate
    within a lineup, so entries are compared on the same cards rather than on
    their luck. Two entries whose intervals overlap share a rank instead of
    being ordered by noise. Each match is also read as its six pairings and
    fitted to an Elo rating, with its own spread from resampling them.

    The ranking figure is **penalty points per deal**, lower being better. A
    reward is this environment's normalisation of those points, so ranking on it
    would make the leaderboard depend on a modelling choice rather than on the
    game.
    """

    if len(entries) < NUM_PLAYERS:
        raise ValueError(f"a table needs {NUM_PLAYERS} entries")
    # Tables are played in chunks so one batch stays a workable size, and every
    # action the rules refuse is counted rather than absorbed.
    width = max(1, min(chunk, lineups))
    rng = np.random.default_rng(seed)
    collected: dict[int, list[float]] = {index: [] for index in range(len(entries))}
    dealt = dict.fromkeys(range(len(entries)), 0)
    won = dict.fromkeys(range(len(entries)), 0)
    lost = dict.fromkeys(range(len(entries)), 0)
    pairings: list[tuple] = []

    for round_index in range(rounds):
        seating = np.stack(
            [rng.permutation(len(entries))[:NUM_PLAYERS] for _ in range(lineups)]
        )
        seating = np.roll(seating, round_index, axis=1)
        keys = jax.random.split(jax.random.key(seed + round_index), lineups)
        pieces = [
            _play_matches(
                entries,
                keys[start : min(start + width, lineups)],
                seating[start : min(start + width, lineups)],
                steps,
            )
            for start in range(0, lineups, width)
        ]
        refused = sum(piece[4] for piece in pieces)
        if refused:
            raise RuntimeError(
                f"the rollout offered {refused} actions the rules refused;"
                " the standings would be meaningless"
            )
        scores, deals, winners, over = (
            np.concatenate([piece[index] for piece in pieces]) for index in range(4)
        )
        if not over.all():
            raise RuntimeError(
                f"{int((~over).sum())} matches did not finish in {steps} events"
            )
        per_deal = np.divide(scores, np.maximum(deals, 1)[:, None], dtype=np.float64)
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
            len(entries), *(part[picks] for part in (left, right, outcome)), passes=120
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
    ranked: list[Standing] = []
    for position, standing in enumerate(standings):
        if position == 0:
            rank = 1
        else:
            previous = ranked[-1]
            gap = standing.points - previous.points
            spread_pair = float(np.hypot(previous.stderr, standing.stderr))
            rank = previous.rank if gap <= spread_pair else position + 1
        ranked.append(replace(standing, rank=rank))
    return ranked
