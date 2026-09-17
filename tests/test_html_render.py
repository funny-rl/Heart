from __future__ import annotations

import jax
import numpy as np
import pytest

import heart


def test_html_player_view_hides_other_hands_and_identifies_viewer():
    env = heart.make()
    state, _ = env.reset(jax.random.key(201))
    viewer = (int(state.active_player) + 1) % heart.NUM_PLAYERS
    rendered = heart.render_html(state, viewer=viewer)

    assert '<main class="table"' in rendered
    assert rendered.count('class="card ') == 52
    assert f"P{viewer} · 나" in rendered
    assert "legal" in rendered

    own_cards = np.flatnonzero(np.asarray(state.hands)[viewer])
    for card in own_cards:
        assert f'aria-label="{heart.card_name(int(card))}"' in rendered

    # Only the viewer's hand is face up; the other three are backs.
    assert rendered.count('aria-label="뒷면"') == 52 - len(own_cards)
    opponent_cards = np.flatnonzero(
        np.asarray(state.hands)[(viewer + 1) % heart.NUM_PLAYERS]
    )
    for card in opponent_cards:
        assert f'aria-label="{heart.card_name(int(card))}"' not in rendered


def test_html_spectator_view_reveals_all_hands_without_viewer_marker():
    env = heart.make()
    state, _ = env.reset(jax.random.key(203))
    rendered = heart.render_html(state, viewer=None)
    assert rendered.count('aria-label="P') >= 4
    assert rendered.count('class="card ') == 52
    assert 'aria-label="뒷면"' not in rendered
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

    output = heart.save_replay_html(states, tmp_path / "replay.html", viewer=2, fps=3.0)
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


def _advance_states(count: int):
    env = heart.make()
    state, observation = env.reset(jax.random.key(401))
    states = [state]
    for _ in range(count):
        action = int(np.flatnonzero(np.asarray(observation.action_mask))[0])
        state, observation, *_ = env.step(state, action)
        states.append(state)
    return states


def test_html_shows_completed_trick_at_fourth_and_terminal_boundaries():
    states = _advance_states(52)
    viewer = 2

    for state, history_index in ((states[4], 0), (states[52], 12)):
        rendered = heart.render_html(state, viewer=viewer)
        cards = np.asarray(state.trick_history)[history_index]
        winner = int(np.asarray(state.trick_winners)[history_index])
        led_suit = int(cards[0]) // 13
        ranks = [
            int(card) % 13 if int(card) // 13 == led_suit else -1 for card in cards
        ]
        leader = (winner - int(np.argmax(ranks))) % heart.NUM_PLAYERS

        assert rendered.count('class="trick-card') == 4
        assert 'aria-label="직전 트릭"' in rendered
        for offset in range(heart.NUM_PLAYERS):
            player = (leader + offset) % heart.NUM_PLAYERS
            seat = ("bottom", "left", "top", "right")[
                (player - viewer) % heart.NUM_PLAYERS
            ]
            assert f'class="trick-card trick-{seat}"><i>P{player}</i>' in rendered

    current = heart.render_html(states[5], viewer=viewer)
    assert current.count('class="trick-card') == 1
    assert 'aria-label="현재 트릭"' in current


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), -float("inf"), 1000.1])
def test_replay_rejects_nonfinite_or_excessive_fps(fps):
    state, _ = heart.make().reset(jax.random.key(403))
    with pytest.raises(ValueError, match="finite"):
        heart.render_replay_html([state], fps=fps)


def test_replay_clamps_fastest_supported_rate_to_one_millisecond():
    state, _ = heart.make().reset(jax.random.key(405))
    rendered = heart.render_replay_html([state], fps=1000)
    assert "const baseInterval=1;" in rendered


def test_html_labels_live_penalties_and_terminal_scores_and_rewards():
    live = heart.render_html(_advance_states(4)[-1])
    assert live.count("누적 벌점") == heart.NUM_PLAYERS
    assert "정산 점수" not in live

    final = heart.render_html(_advance_states(52)[-1])
    assert final.count("정산 점수") == heart.NUM_PLAYERS
    assert final.count('class="reward"') == heart.NUM_PLAYERS
    assert "보상 +" in final or "보상 -" in final
