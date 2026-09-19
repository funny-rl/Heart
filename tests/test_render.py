import jax
import numpy as np

import heart


def test_player_renderer_hides_opponent_cards_and_identifies_viewer():
    env = heart.DealEnv()
    state, _ = env.reset(jax.random.key(0))
    rendered = heart.render_ansi(state, viewer=0)
    assert "P0 (YOU):" in rendered
    assert "hidden cards" not in rendered
    for player in range(heart.NUM_PLAYERS):
        assert f"P{player}" in rendered


def test_ansi_defaults_to_player_zero_view():
    state, _ = heart.DealEnv().reset(jax.random.key(1))
    assert heart.render_ansi(state) == heart.render_ansi(state, viewer=0)
    assert heart.render_ansi(state) != heart.render_ansi(state, viewer=None)


def _advance_state(count: int):
    env = heart.DealEnv()
    state, observation = env.reset(jax.random.key(413))
    for _ in range(count):
        action = observation.action_mask.argmax()
        state, observation, *_ = env.step(state, action)
    return state


def test_ansi_shows_completed_trick_at_fourth_and_terminal_boundaries():
    for step, index in ((4, 0), (52, 12)):
        state = _advance_state(step)
        rendered = heart.render_ansi(state)
        cards = np.asarray(state.trick_history)[index]
        names = " ".join(heart.card_name(int(card)) for card in cards)
        assert f"Last trick: {names}" in rendered

    current = heart.render_ansi(_advance_state(5))
    assert "Table:" in current
    assert "Last trick:" not in current


def test_ansi_labels_live_penalties_and_terminal_scores_and_rewards():
    live = heart.render_ansi(_advance_state(4))
    assert "Penalties:" in live
    assert "Effective scores:" not in live

    final = heart.render_ansi(_advance_state(52))
    assert "Penalties:" in final
    assert "Effective scores:" in final
    assert "Terminal rewards:" in final
