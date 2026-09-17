from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import heart
from heart import contest
from heart.cards import NUM_CARDS
from heart.classic import NUM_PASS_ACTIONS, PASS
from heart.contest import (
    OBSERVATION_DIM,
    OFFSETS,
    SubmissionError,
    encode_observation,
    load_submission,
)

export = pytest.importorskip("jax.export")


def _spec():
    (batch,) = export.symbolic_shape("b")
    return jax.ShapeDtypeStruct((batch, OBSERVATION_DIM), jnp.float32)


def _blob(function, spec=None):
    return export.export(jax.jit(function))(
        _spec() if spec is None else spec
    ).serialize()


def _reference(observations):
    hidden = jnp.tanh(observations @ jnp.full((OBSERVATION_DIM, 32), 0.01))
    return (
        hidden @ jnp.full((32, NUM_PASS_ACTIONS), 0.02),
        hidden @ jnp.full((32, NUM_CARDS), 0.03),
    )


def _playing_state():
    env = heart.make("classic-v0")
    state, _ = env.reset(jax.random.key(11))
    policy = heart.make_rule_pass_policy("medium")
    key = jax.random.key(3)
    while int(state.phase) == PASS:
        key, action_key = jax.random.split(key)
        observation = heart.classic.observe_classic(state, int(state.active_player))
        state = env.step(state, jnp.asarray(int(policy(observation, action_key))))[0]
    return state


def test_layout_covers_the_declared_width():
    assert sum(end - start for start, end in OFFSETS.values()) == OBSERVATION_DIM
    ends = [end for _, end in OFFSETS.values()]
    starts = [start for start, _ in OFFSETS.values()]
    assert starts[0] == 0 and ends[-1] == OBSERVATION_DIM
    assert starts[1:] == ends[:-1], "segments must be contiguous"


def test_observation_describes_the_seat_to_act():
    state = _playing_state()
    player = int(state.active_player)
    encoded = np.asarray(jax.jit(encode_observation)(state, player))

    assert encoded.shape == (OBSERVATION_DIM,)
    assert np.isfinite(encoded).all()
    hand = encoded[slice(*OFFSETS["hand"])]
    legal = encoded[slice(*OFFSETS["legal"])]
    held = np.asarray(state.game.hands)[player]
    np.testing.assert_array_equal(hand, held.astype(np.float32))
    assert legal.sum() >= 1
    assert (legal <= hand).all(), "a legal card must be one the seat holds"
    phase = encoded[slice(*OFFSETS["phase"])]
    np.testing.assert_array_equal(phase, np.asarray([0.0, 1.0], np.float32))


def test_a_well_formed_entry_is_accepted():
    entry = load_submission(_blob(_reference), "reference")
    assert entry.name == "reference"
    assert entry.size_bytes > 0
    assert 0 < entry.flops_per_decision < 5_000_000
    passes, plays = entry.call(jnp.zeros((3, OBSERVATION_DIM), jnp.float32))
    assert passes.shape == (3, NUM_PASS_ACTIONS)
    assert plays.shape == (3, NUM_CARDS)


@pytest.mark.parametrize(
    "function",
    [
        lambda o: (jnp.zeros((o.shape[0], 99)), jnp.zeros((o.shape[0], NUM_CARDS))),
        lambda o: (
            jnp.zeros((o.shape[0], NUM_PASS_ACTIONS), jnp.float16),
            jnp.zeros((o.shape[0], NUM_CARDS)),
        ),
        lambda o: (
            jnp.full((o.shape[0], NUM_PASS_ACTIONS), jnp.nan),
            jnp.zeros((o.shape[0], NUM_CARDS)),
        ),
    ],
    ids=["wrong-shape", "wrong-dtype", "not-finite"],
)
def test_malformed_outputs_are_refused(function):
    with pytest.raises(SubmissionError):
        load_submission(_blob(function), "bad")


def test_wrong_observation_width_is_refused():
    (batch,) = export.symbolic_shape("b")
    narrow = jax.ShapeDtypeStruct((batch, 64), jnp.float32)
    blob = _blob(
        lambda o: (
            jnp.zeros((o.shape[0], NUM_PASS_ACTIONS)),
            jnp.zeros((o.shape[0], NUM_CARDS)),
        ),
        narrow,
    )
    with pytest.raises(SubmissionError, match="input must be"):
        load_submission(blob, "narrow")


def test_an_oversized_computation_is_refused():
    def heavy(observations):
        hidden = observations @ jnp.full((OBSERVATION_DIM, 512), 0.001)
        for _ in range(40):
            hidden = jnp.tanh(hidden @ jnp.full((512, 512), 0.001))
        return (
            hidden @ jnp.full((512, NUM_PASS_ACTIONS), 0.01),
            hidden @ jnp.full((512, NUM_CARDS), 0.01),
        )

    with pytest.raises(SubmissionError, match="budget"):
        load_submission(_blob(heavy), "heavy")


@pytest.mark.parametrize(
    "payload", ["not bytes", b"\x00\x01\x02garbage"], ids=["text", "garbage"]
)
def test_unreadable_payloads_are_refused(payload):
    with pytest.raises(SubmissionError):
        load_submission(payload, "junk")


def test_size_limit_is_enforced_before_parsing():
    with pytest.raises(SubmissionError, match="over"):
        load_submission(b"\x00" * 100, "big", max_bytes=50)


def test_a_host_callback_cannot_be_exported_at_all():
    """The submission format has no route back to the host process."""

    def escaping(observations):
        def on_host(values):
            return np.asarray(values)

        leaked = jax.pure_callback(
            on_host,
            jax.ShapeDtypeStruct((observations.shape[0], NUM_CARDS), jnp.float32),
            observations[:, :NUM_CARDS],
        )
        return jnp.zeros((observations.shape[0], NUM_PASS_ACTIONS)), leaked

    # Export refuses it; the precise error depends on where the lowering trips.
    with pytest.raises(Exception) as refusal:
        _blob(escaping)
    assert not isinstance(refusal.value, AssertionError)


def _entry(name, seed, scale=0.1):
    key_one, key_two, key_three = jax.random.split(jax.random.key(seed), 3)
    first = jax.random.normal(key_one, (OBSERVATION_DIM, 16)) * scale
    second = jax.random.normal(key_two, (16, NUM_PASS_ACTIONS)) * scale
    third = jax.random.normal(key_three, (16, NUM_CARDS)) * scale

    def policy(observations):
        hidden = jnp.tanh(observations @ first)
        return hidden @ second, hidden @ third

    return load_submission(_blob(policy), name)


def test_the_baseline_entry_satisfies_its_own_contract():
    entry = load_submission(contest.baseline_blob(), "baseline")
    assert entry.flops_per_decision < contest.MAX_FLOPS_PER_DECISION
    passes, plays = entry.call(jnp.zeros((5, OBSERVATION_DIM), jnp.float32))
    assert passes.shape == (5, NUM_PASS_ACTIONS)
    assert plays.shape == (5, NUM_CARDS)


def test_a_league_ranks_every_entry_on_the_scores_it_deals():
    entries = [
        load_submission(contest.baseline_blob(), "baseline"),
        _entry("alpha", 1),
        _entry("beta", 2),
        _entry("gamma", 3),
    ]
    table = contest.run_league(entries, lineups=32, rounds=2, seed=5, bootstrap=3)

    assert {standing.name for standing in table} == {e.name for e in entries}
    assert [s.rank for s in table] == sorted(s.rank for s in table)
    assert all(standing.tables > 0 for standing in table)
    # Each table settles several deals, so deals outnumber seatings.
    assert all(standing.deals > standing.tables for standing in table)
    # A deal hands out 26 points between four seats, so the field averages 6.5;
    # moon shots pay 78 and pull it a little higher.
    average = sum(standing.points for standing in table) / len(table)
    assert 6.0 < average < 8.0
    assert all(0.0 <= standing.points < 26.0 for standing in table)
    # Lower is better, so the table reads upward in points.
    assert [s.points for s in table] == sorted(s.points for s in table)
    # Ratings are centred on the starting rating by construction.
    rating = sum(standing.elo for standing in table) / len(table)
    assert abs(rating - contest.START_RATING) < 1e-6
    assert all(standing.elo_stderr >= 0 for standing in table)


def test_a_league_needs_a_full_table():
    with pytest.raises(ValueError, match="table"):
        contest.run_league([_entry("solo", 9)], lineups=8, rounds=1)


def test_ratings_follow_the_pairwise_record():
    """A seat that always finishes ahead must rate above one that never does."""

    left = np.array([0, 0, 0, 1, 1, 2])
    right = np.array([1, 2, 3, 2, 3, 3])
    outcome = np.ones(6, np.float64)  # the lower index always takes the pairing
    ratings = contest._fit_elo(4, left, right, outcome)
    assert list(np.argsort(-ratings)) == [0, 1, 2, 3]
