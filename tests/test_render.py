import jax

import heart


def test_player_renderer_reveals_all_cards_and_identifies_viewer():
    env = heart.make()
    state, _ = env.reset(jax.random.key(0))
    rendered = heart.render_ansi(state, viewer=0)
    assert "P0 (YOU):" in rendered
    assert "hidden cards" not in rendered
    for player in range(heart.NUM_PLAYERS):
        assert f"P{player}" in rendered
