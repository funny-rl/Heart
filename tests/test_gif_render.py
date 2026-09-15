from __future__ import annotations

import jax
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
