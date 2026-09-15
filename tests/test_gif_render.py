from __future__ import annotations

import jax
import numpy as np
import pytest

import heart

Image = pytest.importorskip("PIL.Image")


def test_gif_frame_is_full_size_rgb_image():
    state, _ = heart.make().reset(jax.random.key(301))
    frame = heart.render_gif_frame(state, viewer=2)
    assert frame.size == (960, 540)
    assert frame.mode == "RGB"


def test_save_gif_writes_all_frames(tmp_path):
    env = heart.make()
    state, observation = env.reset(jax.random.key(303))
    next_state, *_ = env.step(state, observation.action_mask.argmax())
    output = heart.save_gif(
        [state, next_state], tmp_path / "preview.gif", duration_ms=100
    )

    with Image.open(output) as rendered:
        assert rendered.n_frames == 2
        assert rendered.size == (960, 540)


def test_gif_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        heart.save_gif([], "unused.gif")
    state, _ = heart.make().reset(jax.random.key(305))
    with pytest.raises(ValueError):
        heart.save_gif([state], "unused.gif", duration_ms=0)
    with pytest.raises(ValueError):
        heart.render_gif_frame(state, viewer=4)


def _advance_state(count: int):
    env = heart.make()
    state, observation = env.reset(jax.random.key(407))
    for _ in range(count):
        action = observation.action_mask.argmax()
        state, observation, *_ = env.step(state, action)
    return state


@pytest.mark.parametrize(("step", "history_index"), [(4, 0), (52, 12)])
def test_gif_frame_draws_completed_trick_at_boundaries(
    monkeypatch, step, history_index
):
    import heart.gif_render as renderer

    state = _advance_state(step)
    drawn = []

    def record(draw, card, xy, size, fonts, *, legal=False):
        drawn.append(int(card))

    monkeypatch.setattr(renderer, "_card", record)
    renderer.render_gif_frame(state)

    hand_count = int(np.asarray(state.hands).sum())
    expected = [int(card) for card in np.asarray(state.trick_history)[history_index]]
    assert drawn[hand_count:] == expected


def test_gif_letterboxes_arbitrary_aspect_ratio():
    state, _ = heart.make().reset(jax.random.key(409))
    standard = np.asarray(heart.render_gif_frame(state))
    wide = np.asarray(heart.render_gif_frame(state, size=(1200, 540)))
    assert np.array_equal(wide[:, 120:1080], standard)
    assert np.all(wide[:, :120] == np.asarray([17, 19, 26]))
    assert np.all(wide[:, 1080:] == np.asarray([17, 19, 26]))


def test_save_gif_requires_gif_suffix_and_forces_format(tmp_path):
    state, _ = heart.make().reset(jax.random.key(411))
    with pytest.raises(ValueError, match=r"\.gif suffix"):
        heart.save_gif([state], tmp_path / "preview.png")

    output = heart.save_gif([state], tmp_path / "preview.GIF")
    assert output.read_bytes().startswith(b"GIF")


def test_gif_labels_live_penalties_and_terminal_scores_and_rewards(monkeypatch):
    import heart.gif_render as renderer

    labels = []
    original_center = renderer._center

    def record(draw, xy, text, font, fill):
        labels.append(text)
        return original_center(draw, xy, text, font, fill)

    monkeypatch.setattr(renderer, "_center", record)
    renderer.render_gif_frame(_advance_state(4))
    assert sum(str(label).startswith("PENALTY ") for label in labels) == 4

    labels.clear()
    renderer.render_gif_frame(_advance_state(52))
    assert sum(str(label).startswith("SCORE ") for label in labels) == 4
    assert all("REWARD" in label for label in labels if str(label).startswith("SCORE "))
