from __future__ import annotations

import jax
import numpy as np
import pytest

import heart


def test_html_player_view_reveals_every_hand_and_identifies_viewer():
    env = heart.make()
    state, _ = env.reset(jax.random.key(201))
    viewer = (int(state.active_player) + 1) % heart.NUM_PLAYERS
    rendered = heart.render_html(state, viewer=viewer)

    assert '<main class="table"' in rendered
    assert rendered.count('class="card ') == 52
    assert f"P{viewer} · 나" in rendered
    assert "legal" in rendered

    opponent_cards = np.flatnonzero(
        np.asarray(state.hands)[(viewer + 1) % heart.NUM_PLAYERS]
    )
    for card in opponent_cards:
        assert f'aria-label="{heart.card_name(int(card))}"' in rendered


def test_html_spectator_view_reveals_all_hands_without_viewer_marker():
    env = heart.make()
    state, _ = env.reset(jax.random.key(203))
    rendered = heart.render_html(state, viewer=None)
    assert rendered.count('aria-label="P') >= 4
    assert rendered.count('class="card ') == 52
    assert "· 나" not in rendered


def test_save_html_writes_standalone_page(tmp_path):
    env = heart.make()
    state, _ = env.reset(jax.random.key(205))
    output = heart.save_html(state, tmp_path / "game.html", viewer=0)
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_replay_html_contains_frames_and_playback_controls(tmp_path):
    env = heart.make("simplest-v0")
    state, observation = env.reset(jax.random.key(207))
    states = [state]
    for _ in range(3):
        action = np.flatnonzero(np.asarray(observation.action_mask))[0]
        state, observation, *_ = env.step(state, action)
        states.append(state)

    output = heart.save_replay_html(
        states, tmp_path / "replay.html", viewer=2, fps=3.0
    )
    rendered = output.read_text(encoding="utf-8")
    assert '<iframe id="stage"' in rendered
    assert 'id="timeline"' in rendered
    assert 'id="speed"' in rendered
    assert 'max="3"' in rendered
    assert "P2 · 나" in rendered


def test_replay_rejects_empty_or_nonpositive_fps():
    with pytest.raises(ValueError):
        heart.render_replay_html([])
    env = heart.make()
    state, _ = env.reset(jax.random.key(209))
    with pytest.raises(ValueError):
        heart.render_replay_html([state], fps=0)
