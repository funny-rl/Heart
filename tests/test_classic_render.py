from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import pytest

import heart


@pytest.fixture(scope="module")
def classic_state():
    state, _ = heart.make("classic-v0").reset(jax.random.key(701))
    pass_cards = jnp.asarray(
        [[0, 1, 2], [13, 14, 15], [-1, -1, -1], [-1, -1, -1]],
        dtype=jnp.int8,
    )
    return state._replace(
        active_player=jnp.asarray(2, dtype=jnp.int32),
        pass_cards=pass_cards,
        match_scores=jnp.asarray([12, 24, 36, 48], dtype=jnp.int16),
    )


def _boundary_state(state, *, hold: bool = False):
    direction = heart.PASS_HOLD if hold else heart.PASS_RIGHT
    phase = heart.PLAY if hold else heart.PASS
    deal_index = 3 if hold else 1
    return state._replace(
        deal_index=jnp.asarray(deal_index, dtype=jnp.int8),
        phase=jnp.asarray(phase, dtype=jnp.int8),
        pass_direction=jnp.asarray(direction, dtype=jnp.int8),
        deal_boundary=jnp.asarray(True),
        last_deal_scores=jnp.asarray([1, 2, 3, 20], dtype=jnp.int16),
        last_deal_rewards=jnp.asarray([1.0, 0.5, 0.0, -1.5], dtype=jnp.float32),
        last_deal_moon_shooter=jnp.asarray(-1, dtype=jnp.int8),
    )


def _terminal_state(state):
    game = state.game._replace(
        terminated=jnp.asarray(True),
        scores=jnp.asarray([0, 26, 10, 10], dtype=jnp.int16),
        penalties=jnp.asarray([26, 0, 0, 0], dtype=jnp.int16),
        winner_mask=jnp.asarray([True, False, False, False]),
    )
    return state._replace(
        game=game,
        deal_index=jnp.asarray(4, dtype=jnp.int8),
        phase=jnp.asarray(heart.TERMINAL, dtype=jnp.int8),
        match_scores=jnp.asarray([101, 80, 110, 120], dtype=jnp.int16),
        last_deal_scores=game.scores,
        last_deal_rewards=jnp.asarray([1.0, -1.0, 0.0, 0.0], dtype=jnp.float32),
        deal_boundary=jnp.asarray(True),
        winner_mask=jnp.asarray([False, True, False, False]),
        terminated=jnp.asarray(True),
    )


def test_classic_html_pass_phase_is_full_observability(classic_state):
    rendered = heart.render_classic_html(classic_state, viewer=2)

    assert "왼쪽 패싱" in rendered
    assert "P2 선택" in rendered
    assert "P2 · 나" in rendered
    assert rendered.count('class="card ') == 52
    assert "2♣ · 3♣ · 4♣" in rendered
    assert "2♦ · 3♦ · 4♦" in rendered
    assert rendered.count("선택 대기") == 2
    assert " legal" not in rendered


def test_classic_html_deal_summary_and_pass_flow_do_not_overlap(classic_state):
    rendered = heart.render_classic_html(_boundary_state(classic_state))

    assert "딜 1 정산" in rendered
    assert "P3</b> 20점" in rendered
    assert "-1.50" in rendered
    assert "다음: 오른쪽 패싱" in rendered
    assert "3장 오른쪽 전달" not in rendered
    assert ".deal-summary,.pass-flow{position:absolute" not in rendered


def test_classic_html_hold_boundary_announces_immediate_play(classic_state):
    rendered = heart.render_classic_html(_boundary_state(classic_state, hold=True))

    assert "딜 3 정산" in rendered
    assert "다음: 홀드 · 바로 플레이" in rendered
    assert '<section class="pass-flow">' not in rendered


def test_classic_html_terminal_uses_match_winner_and_classic_rewards(classic_state):
    rendered = heart.render_classic_html(_terminal_state(classic_state), viewer=0)

    assert "경기 종료 · P1 승리" in rendered
    panels = re.findall(
        r'<section class="player ([^"]*)">.*?<strong>P(\d)',
        rendered,
    )
    highlighted = {int(player) for classes, player in panels if "winner" in classes}
    assert highlighted == {1}
    rewards = re.findall(r'<span class="reward">보상 ([+-]\d+\.\d)</span>', rendered)
    assert rewards == ["+1.0", "-1.0", "+0.0", "+0.0"]
    assert "최종 경기 결과" in rendered


@pytest.mark.parametrize("viewer", [-1, 4])
def test_all_classic_renderers_reject_invalid_viewer(classic_state, viewer):
    with pytest.raises(ValueError, match="viewer"):
        heart.render_classic_html(classic_state, viewer=viewer)
    with pytest.raises(ValueError, match="viewer"):
        heart.render_classic_ansi(classic_state, viewer=viewer)
    pytest.importorskip("PIL.Image")
    with pytest.raises(ValueError, match="viewer"):
        heart.render_classic_gif_frame(classic_state, viewer=viewer)


def test_classic_ansi_pass_and_deal_summary_semantics(classic_state):
    passing = heart.render_classic_ansi(classic_state, viewer=2)
    assert "왼쪽 패싱" in passing
    assert "P2 선택" in passing
    assert "P2 (YOU)" in passing
    assert "Pass choices:" in passing
    assert "P0=2♣ 3♣ 4♣" in passing

    boundary = heart.render_classic_ansi(_boundary_state(classic_state))
    assert "Last deal: P0=1(+1.00)" in boundary
    assert "P3=20(-1.50)" in boundary


def test_classic_ansi_terminal_has_match_winner_and_no_active_arrow(classic_state):
    rendered = heart.render_classic_ansi(_terminal_state(classic_state))

    assert "경기 종료 · P1 승리" in rendered
    assert not any(line.startswith("→") for line in rendered.splitlines())


def test_classic_gif_pass_and_boundary_frames_are_distinct(monkeypatch, classic_state):
    pytest.importorskip("PIL.Image")
    import heart.classic_render as renderer

    texts = []
    original_center = renderer._center

    def record(draw, xy, text, font, fill):
        texts.append((xy, text))
        return original_center(draw, xy, text, font, fill)

    monkeypatch.setattr(renderer, "_center", record)
    renderer.render_classic_gif_frame(_boundary_state(classic_state))

    pass_entries = [(xy, text) for xy, text in texts if text.startswith("PASS RIGHT")]
    deal_entries = [(xy, text) for xy, text in texts if text == "DEAL 1 COMPLETE"]
    assert len(pass_entries) == 0
    assert len(deal_entries) == 1


def test_classic_gif_terminal_names_match_winner(monkeypatch, classic_state):
    pytest.importorskip("PIL.Image")
    import heart.classic_render as renderer

    texts = []

    def record(draw, xy, text, font, fill):
        texts.append(text)

    monkeypatch.setattr(renderer, "_center", record)
    renderer.render_classic_gif_frame(_terminal_state(classic_state))

    assert any("MATCH COMPLETE" in text and "WINNER P1" in text for text in texts)


def test_save_classic_gif_preserves_each_replay_frame(tmp_path, classic_state):
    Image = pytest.importorskip("PIL.Image")
    frames = [
        classic_state,
        _boundary_state(classic_state),
        _terminal_state(classic_state),
    ]

    output = heart.save_classic_gif(frames, tmp_path / "classic.gif", duration_ms=120)

    with Image.open(output) as rendered:
        assert rendered.n_frames == len(frames)
        durations = []
        for index in range(rendered.n_frames):
            rendered.seek(index)
            durations.append(rendered.info["duration"])
        assert durations == [120, 1200, 1200]


def test_classic_replay_html_embeds_event_frames(classic_state):
    rendered = heart.render_classic_replay_html(
        [classic_state, _boundary_state(classic_state)], fps=4
    )

    assert "HEART Replay" in rendered
    assert 'max="1"' in rendered
    assert "딜 1 정산" in rendered
