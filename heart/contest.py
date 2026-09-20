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

import json
import re
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import heart
from heart.cards import HAND_SIZE, NUM_CARDS, NUM_PLAYERS
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
from heart.types import Observation

MAX_SUBMISSION_BYTES = 8 * 1024 * 1024
MAX_FLOPS_PER_DECISION = 5_000_000
MAX_SEED = 2**32 - 1
START_RATING = 1500.0
RATING_SCALE = 400.0
_CUSTOM_CALL = re.compile(
    r'call_target_name\s*=\s*"([^"]+)"'
    r'|stablehlo\.custom_call\s+@(?:"([^"]+)"|([\w.$-]+))'
)
_ALLOWED_CUSTOM_CALLS = frozenset({"shape_assertion"})
_REGISTERED = False


class SubmissionError(ValueError):
    """A submission that the contract refuses to run."""


def _integer_parameter(
    name: str,
    value: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


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


def save_submission(
    policy,
    directory: str | Path,
    *,
    name: str,
    description: str,
    platforms: tuple[str, ...] | None = None,
) -> Submission:
    """Export, validate, and write a ready-to-submit policy directory."""

    values = {}
    for field, value, limit in (
        ("name", name, 64),
        ("description", description, 200),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        if "\n" in value or "\r" in value:
            raise ValueError(f"{field} must fit on one line")
        if len(value.strip()) > limit:
            raise ValueError(f"{field} must be at most {limit} characters")
        values[field] = value.strip()

    blob = export_policy(policy, platforms=platforms)
    entry = load_submission(blob, values["name"])
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "entry.bin": blob,
        "entry.toml": (
            f"name = {json.dumps(values['name'], ensure_ascii=False)}\n"
            f"description = {json.dumps(values['description'], ensure_ascii=False)}\n"
        ).encode(),
        "validation.json": (
            json.dumps(
                {
                    "name": entry.name,
                    "size_bytes": entry.size_bytes,
                    "flops_per_decision": entry.flops_per_decision,
                },
                indent=2,
            )
            + "\n"
        ).encode(),
    }
    for filename, contents in files.items():
        path = destination / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(contents)
        temporary.replace(path)
    return entry


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


def _custom_call_targets(module: str) -> set[str]:
    return {
        next(target for target in match if target)
        for match in _CUSTOM_CALL.findall(module)
    }


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

    targets = sorted(
        _custom_call_targets(exported.mlir_module()) - _ALLOWED_CUSTOM_CALLS
    )
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

    Each match is read as every pairing its seats imply. Sequential Elo
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
    """Every match becomes the seat-versus-seat comparisons its seats imply.

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


def _carry_layouts(
    state: ClassicState, following: ClassicState, pass_hands, play_hands
):
    """Update hand-slot layouts at deal and pass boundaries."""

    fresh = jax.vmap(_slots)(following)
    new_deal = (following.deal_index != state.deal_index)[:, None, None]
    after_pass = ((state.phase == PASS) & (following.phase == PLAY))[
        :, None, None
    ] & ~new_deal
    return (
        jnp.where(new_deal, fresh, pass_hands),
        jnp.where(new_deal | after_pass, fresh, play_hands),
    )


def _league_step(entries: list[Submission], batch: int):
    """One event for a batch of matches, compiled on its own.

    The loop over steps stays in Python so that finished matches can be
    dropped between steps: a terminal match has no legal action left, and
    stepping it would count a refusal against whichever entry sat there.
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

        outputs = [entry.call(observation) for entry in entries]
        passes = jnp.stack([output[0] for output in outputs])
        plays = jnp.stack([output[1] for output in outputs])
        acting = jnp.take_along_axis(seating, players[:, None], 1)[:, 0]
        selected_passes = passes[acting, rows]
        selected_plays = plays[acting, rows]
        nonfinite = ~(
            jnp.all(jnp.isfinite(selected_passes), axis=-1)
            & jnp.all(jnp.isfinite(selected_plays), axis=-1)
        )
        pass_action = jnp.argmax(
            jnp.where(observation.pass_action_mask, selected_passes, -jnp.inf), -1
        )
        slot = jnp.argmax(
            jnp.where(observation.play_action_mask, selected_plays, -jnp.inf), -1
        )
        card = jnp.take_along_axis(mine, slot[:, None], 1)[:, 0].astype(jnp.int32)
        actions = jnp.where(state.phase == PASS, pass_action, card)
        following, _, _, _, info = advance(state, actions)

        carried_pass, carried_play = _carry_layouts(
            state, following, pass_hands, play_hands
        )
        return (
            following,
            carried_pass,
            carried_play,
            info.deal_completed,
            info.invalid_action,
            nonfinite,
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
    nonfinite = 0
    for _ in range(steps):
        # A finished match has no legal action left, so it is neither stepped
        # for its result nor counted when it declines to move.
        alive = np.asarray(jax.device_get(states.phase != TERMINAL))
        if not alive.any():
            break
        states, pass_hands, play_hands, settled, invalid, bad_logits = step(
            states, pass_hands, play_hands, seating
        )
        deals += np.asarray(jax.device_get(settled)) * alive
        refused += int((np.asarray(jax.device_get(invalid)) & alive).sum())
        nonfinite += int((np.asarray(jax.device_get(bad_logits)) & alive).sum())
        if refused or nonfinite:
            break
    return (
        np.asarray(jax.device_get(states.match_scores), np.float64),
        deals,
        np.asarray(jax.device_get(states.winner_mask)),
        np.asarray(jax.device_get(states.phase == 2)),
        refused,
        nonfinite,
    )


def _league_schedule(count: int, lineups: int, rounds: int, seed: int):
    rng = np.random.default_rng(seed)
    schedule = []
    for cycle_start in range(0, rounds, NUM_PLAYERS):
        base_seating = np.stack(
            [rng.permutation(count)[:NUM_PLAYERS] for _ in range(lineups)]
        )
        cycle_key = jax.random.fold_in(jax.random.key(seed), cycle_start // NUM_PLAYERS)
        keys = jax.random.split(cycle_key, lineups)
        cycle_rounds = min(NUM_PLAYERS, rounds - cycle_start)
        schedule.extend(
            (np.roll(base_seating, offset, axis=1), keys)
            for offset in range(cycle_rounds)
        )
    return schedule


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

    Each block of up to four rounds keeps its lineups and deal keys while seats
    rotate, so a complete block puts every entry in every seat on the same
    cards. Adjacent entries whose gap is no larger than their combined standard
    error share a rank instead of being ordered by noise. Each match is also
    read as the pairings its seats imply and fitted to an Elo rating, with its
    own spread from cluster resampling.

    The ranking figure is **penalty points per deal**, lower being better. It is
    reported directly rather than through the environment's sign-flipped reward.
    """

    if len(entries) < NUM_PLAYERS:
        raise ValueError(f"a table needs {NUM_PLAYERS} entries")
    lineups = _integer_parameter("lineups", lineups, minimum=1)
    rounds = _integer_parameter("rounds", rounds, minimum=1)
    steps = _integer_parameter("steps", steps, minimum=1)
    seed = _integer_parameter("seed", seed, minimum=0, maximum=MAX_SEED)
    bootstrap = _integer_parameter("bootstrap", bootstrap, minimum=0)
    chunk = _integer_parameter("chunk", chunk, minimum=1)
    # Tables are played in chunks so one batch stays a workable size, and every
    # action the rules refuse is counted rather than absorbed.
    width = max(1, min(chunk, lineups))
    collected: dict[int, dict[int, list[float]]] = {
        index: {} for index in range(len(entries))
    }
    dealt = dict.fromkeys(range(len(entries)), 0)
    won = dict.fromkeys(range(len(entries)), 0)
    lost = dict.fromkeys(range(len(entries)), 0)
    pairings: list[tuple] = []
    pairing_groups: list[np.ndarray] = []

    schedule = _league_schedule(len(entries), lineups, rounds, seed)
    for round_index, (seating, keys) in enumerate(schedule):
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
        nonfinite = sum(piece[5] for piece in pieces)
        if refused:
            raise RuntimeError(
                f"the rollout offered {refused} actions the rules refused;"
                " the standings would be meaningless"
            )
        if nonfinite:
            raise RuntimeError(
                f"policies returned non-finite logits for {nonfinite} live decisions"
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
        groups = (round_index // NUM_PLAYERS) * lineups + np.arange(lineups)
        pairing_groups.append(np.tile(groups, 6))
        for seat in range(NUM_PLAYERS):
            for index in range(len(entries)):
                chosen = seating[:, seat] == index
                if chosen.any():
                    for row in np.flatnonzero(chosen):
                        collected[index].setdefault(int(groups[row]), []).append(
                            float(per_deal[row, seat])
                        )
                    won[index] += int(winners[chosen, seat].sum())
                    lost[index] += int(losers[chosen, seat].sum())
                    dealt[index] += int(deals[chosen].sum())

    left = np.concatenate([pair[0] for pair in pairings])
    right = np.concatenate([pair[1] for pair in pairings])
    outcome = np.concatenate([pair[2] for pair in pairings])
    groups = np.concatenate(pairing_groups)
    ratings = _fit_elo(len(entries), left, right, outcome)
    generator = np.random.default_rng(seed + 1)
    group_order = np.argsort(groups, kind="stable")
    _, group_starts, group_sizes = np.unique(
        groups[group_order], return_index=True, return_counts=True
    )
    group_indices = [
        group_order[start : start + size]
        for start, size in zip(group_starts, group_sizes, strict=True)
    ]
    resampled = [
        _fit_elo(
            len(entries), *(part[picks] for part in (left, right, outcome)), passes=120
        )
        for picks in (
            np.concatenate(
                [
                    group_indices[index]
                    for index in generator.integers(
                        0, len(group_indices), len(group_indices)
                    )
                ]
            )
            for _ in range(bootstrap)
        )
    ]
    spread = (
        np.stack(resampled).std(axis=0, ddof=1)
        if len(resampled) > 1
        else np.zeros(len(entries))
    )

    standings = []
    for index, entry in enumerate(entries):
        cluster_values = np.asarray(
            [np.mean(values) for values in collected[index].values()], np.float64
        )
        matches = sum(len(values) for values in collected[index].values())
        mean = float(cluster_values.mean()) if cluster_values.size else 0.0
        error = (
            float(cluster_values.std(ddof=1) / np.sqrt(cluster_values.size))
            if cluster_values.size > 1
            else 0.0
        )
        standings.append(
            Standing(
                0,
                entry.name,
                mean,
                error,
                won[index] / max(matches, 1),
                lost[index] / max(matches, 1),
                float(ratings[index]),
                float(spread[index]),
                matches,
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
