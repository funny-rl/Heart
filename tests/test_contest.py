from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import heart
from heart import contest
from heart.cards import HAND_SIZE
from heart.classic import NUM_PASS_ACTIONS
from heart.contest import (
    SubmissionError,
    export_policy,
    load_submission,
    observation_signature,
    sample_observation,
)
from submissions import validate

export = pytest.importorskip("jax.export")


def _reference(observation):
    hidden = jnp.tanh(
        observation.play_hand_cards.astype(jnp.float32)
        @ jnp.full((HAND_SIZE, 16), 0.01)
    )
    return (
        hidden @ jnp.full((16, NUM_PASS_ACTIONS), 0.02),
        hidden @ jnp.full((16, HAND_SIZE), 0.03),
    )


def _entry(name, seed, scale=0.3):
    keys = jax.random.split(jax.random.key(seed), 3)
    first = jax.random.normal(keys[0], (HAND_SIZE, 16)) * scale
    second = jax.random.normal(keys[1], (16, NUM_PASS_ACTIONS)) * scale
    third = jax.random.normal(keys[2], (16, HAND_SIZE)) * scale

    def policy(observation):
        hidden = jnp.tanh(observation.play_hand_cards.astype(jnp.float32) @ first)
        return hidden @ second, hidden @ third

    return load_submission(export_policy(policy), name)


def test_the_signature_is_the_environment_s_own_observation():
    """A second encoding would be a second thing to keep true."""

    sample = sample_observation()
    signature = observation_signature()
    assert jax.tree.structure(signature) == jax.tree.structure(sample)
    for spec, leaf in zip(jax.tree.leaves(signature), jax.tree.leaves(sample)):
        assert spec.dtype == leaf.dtype
        assert [str(dim) for dim in spec.shape[1:]] == [str(d) for d in leaf.shape]


def test_a_well_formed_entry_is_accepted():
    entry = load_submission(export_policy(_reference), "reference")
    assert entry.size_bytes > 0
    assert 0 < entry.flops_per_decision < contest.MAX_FLOPS_PER_DECISION
    batch = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (3, *x.shape)), sample_observation()
    )
    passes, plays = entry.call(batch)
    assert passes.shape == (3, NUM_PASS_ACTIONS)
    assert plays.shape == (3, HAND_SIZE)


@pytest.mark.parametrize(
    "policy",
    [
        lambda o: (
            jnp.zeros((o.play_hand_cards.shape[0], 99)),
            jnp.zeros((o.play_hand_cards.shape[0], HAND_SIZE)),
        ),
        lambda o: (
            jnp.zeros((o.play_hand_cards.shape[0], NUM_PASS_ACTIONS), jnp.float16),
            jnp.zeros((o.play_hand_cards.shape[0], HAND_SIZE)),
        ),
        lambda o: (
            jnp.full((o.play_hand_cards.shape[0], NUM_PASS_ACTIONS), jnp.nan),
            jnp.zeros((o.play_hand_cards.shape[0], HAND_SIZE)),
        ),
    ],
    ids=["wrong-shape", "wrong-dtype", "not-finite"],
)
def test_malformed_outputs_are_refused(policy):
    with pytest.raises(SubmissionError):
        load_submission(export_policy(policy), "bad")


def test_a_wrong_observation_is_refused():
    """An entry built against anything but the contract cannot be seated."""

    batch = export.symbolic_shape("b")[0]
    narrow = jax.ShapeDtypeStruct((batch, 64), jnp.float32)
    blob = export.export(
        jax.jit(
            lambda o: (
                jnp.zeros((o.shape[0], NUM_PASS_ACTIONS)),
                jnp.zeros((o.shape[0], HAND_SIZE)),
            )
        ),
        platforms=contest.host_platforms(),
    )(narrow).serialize()
    with pytest.raises(SubmissionError, match="observation"):
        load_submission(blob, "narrow")


def test_an_oversized_computation_is_refused():
    def heavy(observation):
        hidden = observation.play_hand_cards.astype(jnp.float32) @ jnp.full(
            (HAND_SIZE, 512), 0.001
        )
        for _ in range(40):
            hidden = jnp.tanh(hidden @ jnp.full((512, 512), 0.001))
        return (
            hidden @ jnp.full((512, NUM_PASS_ACTIONS), 0.01),
            hidden @ jnp.full((512, HAND_SIZE), 0.01),
        )

    with pytest.raises(SubmissionError, match="budget"):
        load_submission(export_policy(heavy), "heavy")


def test_an_entry_for_another_platform_is_refused():
    blob = export_policy(_reference, platforms=("tpu",))
    with pytest.raises(SubmissionError, match="platform|host"):
        load_submission(blob, "elsewhere")


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

    def escaping(observation):
        leaked = jax.pure_callback(
            lambda values: np.asarray(values),
            jax.ShapeDtypeStruct(
                (observation.play_hand_cards.shape[0], HAND_SIZE), jnp.float32
            ),
            observation.play_hand_cards.astype(jnp.float32),
        )
        return jnp.zeros(
            (observation.play_hand_cards.shape[0], NUM_PASS_ACTIONS)
        ), leaked

    with pytest.raises(Exception) as refusal:
        export_policy(escaping)
    assert not isinstance(refusal.value, AssertionError)


def test_custom_call_scanner_understands_current_and_legacy_mlir():
    module = """
      stablehlo.custom_call @shape_assertion(%0) {has_side_effect = true}
      stablehlo.custom_call @"unsafe.target"(%1)
      custom_call @legacy {call_target_name = "legacy_target"}
    """
    assert contest._custom_call_targets(module) == {
        "shape_assertion",
        "unsafe.target",
        "legacy_target",
    }


def test_the_baseline_entry_satisfies_its_own_contract():
    entry = load_submission(contest.baseline_blob(), "baseline")
    assert entry.flops_per_decision < contest.MAX_FLOPS_PER_DECISION


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
    assert all(standing.matches > 0 for standing in table)
    # A match runs to 100 points, which takes several deals.
    assert all(standing.deals > 4 * standing.matches for standing in table)
    # A deal hands out 26 points between four seats, so the field averages 6.5;
    # moon shots pay 78 and pull it a little higher.
    average = sum(standing.points for standing in table) / len(table)
    assert 5.0 < average < 9.0
    assert sum(standing.win_rate for standing in table) >= 1.0
    assert sum(standing.last_rate for standing in table) >= 1.0
    assert [s.points for s in table] == sorted(s.points for s in table)
    assert table[0].win_rate >= table[-1].win_rate
    assert table[0].last_rate <= table[-1].last_rate
    rating = sum(standing.elo for standing in table) / len(table)
    assert abs(rating - contest.START_RATING) < 1e-6


def test_league_refuses_logits_that_become_nonfinite_after_the_probe():
    def policy(observation):
        batch = observation.play_hand_cards.shape[0]
        fail = observation.match.game.trick_index > 0
        passes = jnp.zeros((batch, NUM_PASS_ACTIONS), dtype=jnp.float32)
        plays = jnp.where(
            fail[:, None],
            jnp.full((batch, HAND_SIZE), jnp.nan),
            jnp.zeros((batch, HAND_SIZE), dtype=jnp.float32),
        )
        return passes, plays

    bad = load_submission(export_policy(policy), "later-nan")
    entries = [bad] + [
        load_submission(contest.baseline_blob(), f"baseline-{index}")
        for index in range(3)
    ]
    with pytest.raises(RuntimeError, match="non-finite logits"):
        contest.run_league(entries, lineups=1, rounds=1, seed=8, bootstrap=0, chunk=1)


def test_a_league_needs_a_full_table():
    with pytest.raises(ValueError, match="table"):
        contest.run_league([None], lineups=8, rounds=1)


def test_league_schedule_rotates_shared_deals_before_resampling():
    schedule = contest._league_schedule(7, lineups=5, rounds=8, seed=17)

    assert len(schedule) == 8
    for cycle_start in (0, 4):
        base_seating, base_keys = schedule[cycle_start]
        for offset in range(4):
            seating, keys = schedule[cycle_start + offset]
            np.testing.assert_array_equal(
                seating, np.roll(base_seating, offset, axis=1)
            )
            np.testing.assert_array_equal(
                jax.random.key_data(keys), jax.random.key_data(base_keys)
            )
    assert not np.array_equal(
        jax.random.key_data(schedule[0][1]), jax.random.key_data(schedule[4][1])
    )


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("lineups", 0, ValueError),
        ("rounds", 0, ValueError),
        ("steps", 0, ValueError),
        ("chunk", 0, ValueError),
        ("bootstrap", -1, ValueError),
        ("seed", -1, ValueError),
        ("seed", 2**32, ValueError),
        ("lineups", True, TypeError),
        ("rounds", 1.5, TypeError),
    ],
)
def test_league_parameters_are_validated_before_rollout(name, value, error):
    with pytest.raises(error, match=name):
        contest.run_league([None] * 4, **{name: value})


def test_zero_bootstrap_is_supported():
    entries = [
        load_submission(contest.baseline_blob(), f"baseline-{index}")
        for index in range(4)
    ]
    table = contest.run_league(
        entries,
        lineups=1,
        rounds=1,
        seed=9,
        bootstrap=0,
        chunk=1,
    )
    assert all(standing.elo_stderr == 0.0 for standing in table)


def test_submission_validator_ignores_python_cache(tmp_path, monkeypatch):
    (tmp_path / "__pycache__").mkdir()
    monkeypatch.setattr(validate, "ROOT", tmp_path)
    assert validate.main([]) == 0


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ('name = 3\ndescription = "valid"\n', "name must be a string"),
        ('name = "valid"\ndescription = 3\n', "description must be a string"),
        ('name = "valid"\ndescription = "first\\nsecond"\n', "fit on one line"),
    ],
)
def test_submission_validator_rejects_malformed_manifest(
    tmp_path, monkeypatch, manifest, message
):
    team = tmp_path / "valid-team"
    team.mkdir()
    (team / "entry.bin").write_bytes(b"graph")
    (team / "entry.toml").write_text(manifest)
    monkeypatch.setattr(validate, "ROOT", tmp_path)
    assert validate.main([]) == 1
    assert message in "\n".join(validate.check(team))


def test_submission_validator_checks_size_before_loading(tmp_path, monkeypatch):
    team = tmp_path / "valid-team"
    team.mkdir()
    (team / "entry.bin").write_bytes(b"large")
    (team / "entry.toml").write_text('description = "valid"\n')
    monkeypatch.setattr(validate, "MAX_SUBMISSION_BYTES", 4)
    monkeypatch.setattr(
        validate,
        "load_submission",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not load")),
    )
    assert "entry.bin is over 4 bytes" in "\n".join(validate.check(team))


def test_submission_validator_does_not_reread_oversized_metadata(tmp_path, monkeypatch):
    team = tmp_path / "valid-team"
    team.mkdir()
    (team / "entry.bin").write_bytes(b"graph")
    meta = team / "entry.toml"
    meta.write_text('name = "valid"\ndescription = "valid"\n')
    monkeypatch.setattr(validate, "MAX_METADATA_BYTES", 4)

    assert "entry.toml is over 4 bytes" in "\n".join(validate.check(team))
    assert validate.clashing_names([team]) == []


def test_submission_validator_rejects_non_utf8_metadata(tmp_path):
    team = tmp_path / "valid-team"
    team.mkdir()
    (team / "entry.bin").write_bytes(b"graph")
    (team / "entry.toml").write_bytes(b"\xff")

    assert "entry.toml is not readable" in "\n".join(validate.check(team))
    assert validate.clashing_names([team]) == []


@pytest.mark.parametrize("filename", ["entry.bin", "entry.toml"])
def test_submission_validator_requires_regular_entry_files(tmp_path, filename):
    team = tmp_path / "valid-team"
    team.mkdir()
    (team / "entry.bin").write_bytes(b"graph")
    (team / "entry.toml").write_text('description = "valid"\n')
    (team / filename).unlink()
    (team / filename).mkdir()

    assert "must be regular files" in "\n".join(validate.check(team))


def test_submission_validator_rejects_symbolic_entry_files(tmp_path):
    team = tmp_path / "valid-team"
    team.mkdir()
    target = tmp_path / "graph"
    target.write_bytes(b"graph")
    (team / "entry.bin").symlink_to(target)
    (team / "entry.toml").write_text('description = "valid"\n')
    assert "must not be symbolic links" in "\n".join(validate.check(team))


@pytest.mark.parametrize("name", ["a", "a-b", "abc123", "a" * 39])
def test_submission_team_name_accepts_github_handle_shape(name):
    assert validate.TEAM.fullmatch(name)


@pytest.mark.parametrize("name", ["", "A", "-a", "a-", "a--b", "a" * 40])
def test_submission_team_name_rejects_non_github_handle_shape(name):
    assert not validate.TEAM.fullmatch(name)


def test_ratings_follow_the_pairwise_record():
    left = np.array([0, 0, 0, 1, 1, 2])
    right = np.array([1, 2, 3, 2, 3, 3])
    outcome = np.ones(6, np.float64)  # the lower index always takes the pairing
    ratings = contest._fit_elo(4, left, right, outcome)
    assert list(np.argsort(-ratings)) == [0, 1, 2, 3]


def test_hand_slots_hold_still_while_a_deal_is_played():
    """The action axis must mean the same card from one turn to the next."""

    env = heart.make_classic_single_agent(
        controlled_player=0, pass_opponents="medium", play_opponents="medium"
    )
    step = jax.jit(env.step)
    state, observation = env.reset(jax.random.key(11))
    while int(observation.match.phase) == heart.PASS:
        legal = np.flatnonzero(np.asarray(observation.pass_action_mask))
        state, observation, *_ = step(state, jnp.int32(legal[0]))
    layouts = [np.asarray(observation.play_hand_cards).copy()]
    for _ in range(4):
        legal = np.flatnonzero(np.asarray(observation.play_action_mask))
        if legal.size == 0:
            break
        state, observation, *_ = step(state, jnp.int32(legal[0]))
        layouts.append(np.asarray(observation.play_hand_cards).copy())
    for earlier, later in itertools.pairwise(layouts):
        np.testing.assert_array_equal(earlier, later)


def test_the_league_shows_a_seat_what_the_adapter_shows_a_learner():
    """The contract is the environment's interface only if it reproduces it.

    The league builds observations itself rather than driving the adapter, so
    the two can drift, and they did: the league refreshed the play layout on
    every new deal while the adapter held the previous deal's layout through a
    passing phase. The adapter was the one in the wrong, but either way a
    contract that matches the environment in every phase but one is a second
    convention wearing the first one's name.
    """

    seat = 0
    env = heart.make_single_agent(
        "classic-v0", controlled_player=seat, opponents="easy"
    )
    step = jax.jit(env.step)
    state, observation = env.reset(jax.random.key(7))
    # The league's carry is driven here, so a change to its rule reaches
    # this assertion rather than being restated by it.
    batched = jax.tree.map(lambda leaf: leaf[None], state.match)
    pass_hands = play_hands = contest._slots(state.match)[None]

    boundaries = 0
    for _ in range(200):
        rebuilt = contest._seat_observation(
            state.match, jnp.int32(seat), pass_hands[0, seat], play_hands[0, seat]
        )
        for shown, built in zip(
            jax.tree.leaves(observation), jax.tree.leaves(rebuilt), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(shown), np.asarray(built))

        passing = int(state.match.phase) == heart.PASS
        mask = observation.pass_action_mask if passing else observation.play_action_mask
        before = state.match
        state, observation, _, done, info = step(state, jnp.int32(jnp.argmax(mask)))
        assert not bool(info.core.invalid_action)

        following = jax.tree.map(lambda leaf: leaf[None], state.match)
        pass_hands, play_hands = contest._carry_layouts(
            batched, following, pass_hands, play_hands
        )
        batched = following
        if bool(state.match.deal_index != before.deal_index):
            boundaries += 1
        if bool(done):
            break

    # A single deal would never exercise the refresh rule the drift lived in.
    assert boundaries >= 2
