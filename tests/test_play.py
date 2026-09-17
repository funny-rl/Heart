from __future__ import annotations

import itertools
import json
import re
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

import heart
from heart.classic import PASS, PASS_COMBINATIONS
from heart.play import HumanGame, make_rule_seat_policy, pass_action_for, serve


def _opponents(difficulty: str = "medium", human_seat: int = 0):
    return {
        seat: make_rule_seat_policy(difficulty)
        for seat in range(heart.NUM_PLAYERS)
        if seat != human_seat
    }


def _ready(**kwargs) -> HumanGame:
    """A game advanced to the point where the person is on turn."""

    game = HumanGame(**kwargs)
    game.advance()
    return game


def _human_move(game: HumanGame) -> int:
    if int(game.state.phase) == PASS:
        return pass_action_for([0, 1, 2])
    return game.legal_plays()[0]


def test_pass_action_maps_slots_to_its_combination():
    for action in (0, 1, 143, 285):
        slots = np.asarray(PASS_COMBINATIONS)[action].tolist()
        assert pass_action_for(slots) == action
        assert pass_action_for(list(reversed(slots))) == action


@pytest.mark.parametrize("slots", [[0, 1], [0, 1, 2, 3], [0, 0, 1]])
def test_pass_action_rejects_malformed_selections(slots):
    with pytest.raises(ValueError):
        pass_action_for(slots)


def test_rule_seat_policy_rejects_unknown_tier():
    with pytest.raises(ValueError):
        make_rule_seat_policy("impossible")


def test_load_seat_policy_resolves_tiers_and_plugins():
    assert heart.load_seat_policy("easy") is not None
    plugin = heart.load_seat_policy("heart.play:make_rule_seat_policy(hard)")
    game = _ready(seats={1: plugin, 2: plugin, 3: plugin}, seed=5)
    assert game.waiting_for_human


@pytest.mark.parametrize("spec", ["nope", "heart.play", "heart.play:missing"])
def test_load_seat_policy_rejects_bad_specs(spec):
    with pytest.raises((ValueError, AttributeError)):
        heart.load_seat_policy(spec)


def test_game_requires_a_policy_for_every_other_seat():
    with pytest.raises(ValueError):
        HumanGame(seats={1: make_rule_seat_policy("easy")})
    with pytest.raises(ValueError):
        HumanGame(seats=_opponents(), human_seat=heart.NUM_PLAYERS)


def test_game_stops_on_the_human_turn():
    game = _ready(seats=_opponents(human_seat=2), human_seat=2, seed=11)
    assert game.waiting_for_human
    assert int(game.state.active_player) == 2


def test_advance_once_plays_a_single_seat_and_stops_at_the_human():
    game = HumanGame(seats=_opponents(human_seat=3), human_seat=3, seed=13)
    steps = 0
    while game.advance_once():
        steps += 1
        assert steps <= heart.NUM_PLAYERS
    assert steps >= 1
    assert game.waiting_for_human and not game.pending
    assert len(game.log) == steps
    assert game.advance_once() is False


def test_illegal_human_action_is_refused_and_leaves_state_untouched():
    game = _ready(seats=_opponents(), seed=2)
    while int(game.state.phase) == PASS:
        game.play_human(pass_action_for([0, 1, 2]))
        game.advance()
    legal = set(game.legal_plays())
    illegal = next(card for card in range(heart.NUM_CARDS) if card not in legal)
    before = np.asarray(game.state.game.hands)
    with pytest.raises(ValueError):
        game.play_human(illegal)
    np.testing.assert_array_equal(np.asarray(game.state.game.hands), before)


def test_snapshot_offers_only_legal_cards():
    game = _ready(seats=_opponents(), seed=7)
    while int(game.state.phase) == PASS:
        game.play_human(pass_action_for([0, 1, 2]))
        game.advance()
    snapshot = game.snapshot()
    assert snapshot["your_turn"] and not snapshot["finished"]
    assert not snapshot["pending"]
    assert snapshot["active"] == snapshot["seat"]
    assert 0 <= snapshot["leader"] < heart.NUM_PLAYERS
    held = {card["card"] for card in snapshot["hand"]}
    assert snapshot["legal"] and set(snapshot["legal"]) <= held
    assert snapshot["view"].startswith("<!doctype html>")


def test_match_plays_through_to_a_winner():
    game = _ready(seats=_opponents(), seed=3)
    for _ in range(600):
        if game.finished:
            break
        game.play_human(_human_move(game))
        game.advance()
    assert game.finished
    snapshot = game.snapshot()
    assert snapshot["winners"]
    assert max(snapshot["scores"]) >= 100


def test_page_defines_every_element_its_script_looks_up():
    """A missing element makes the page hang on its first render."""

    from heart.play import PAGE

    wanted = set(re.findall(r"getElementById\('([^']+)'\)", PAGE))
    assert wanted
    missing = [name for name in wanted if f'id="{name}"' not in PAGE]
    assert not missing, f"script reads elements the page never defines: {missing}"


def test_page_calls_every_function_its_script_defines():
    """A helper that is never invoked leaves the table frozen mid-turn."""

    from heart.play import PAGE

    defined = set(re.findall(r"(?:async\s+)?function\s+(\w+)\s*\(", PAGE))
    assert defined
    unused = [name for name in defined if len(re.findall(rf"\b{name}\s*\(", PAGE)) < 2]
    assert not unused, f"script defines but never calls: {unused}"


def test_page_script_has_no_string_broken_across_lines():
    """An unterminated literal kills the whole script, so nothing renders."""

    from heart.play import PAGE

    script = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
    for number, line in enumerate(script.splitlines(), 1):
        without_escapes = re.sub(r"\\.", "", line)
        for quote in ("'", '"'):
            assert without_escapes.count(quote) % 2 == 0, (
                f"line {number} leaves a {quote} string open: {line!r}"
            )


def _request(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"content-type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.read()


def test_server_serves_the_page_and_applies_actions():
    game = _ready(seats=_opponents(), seed=4)
    server = serve(game, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, body = _request(base + "/")
        assert status == 200 and b"<iframe" in body

        status, body = _request(base + "/state")
        snapshot = json.loads(body)
        assert status == 200 and snapshot["your_turn"]
        assert snapshot["phase"] == PASS and len(snapshot["hand"]) == 13

        status, body = _request(base + "/action", {"slots": [0, 1, 2]})
        after = json.loads(body)
        assert status == 200 and after["seat"] == 0
        # The human action alone; the table is paced by the page from here.
        assert after["pending"] and not after["your_turn"]

        status, body = _request(base + "/advance", {})
        stepped = json.loads(body)
        assert status == 200
        assert len(stepped["log"]) == len(after["log"]) + 1

        # A batched advance returns one frame per action, in order.
        status, body = _request(base + "/advance", {"steps": 8})
        frames = json.loads(body)["frames"]
        assert status == 200 and frames
        for earlier, later in itertools.pairwise(frames):
            assert len(later["log"]) == len(earlier["log"]) + 1
        assert frames[-1]["your_turn"] or frames[-1]["finished"]

        with pytest.raises(urllib.error.HTTPError) as bad_steps:
            _request(base + "/advance", {"steps": 0})
        assert bad_steps.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as refused:
            _request(base + "/action", {"slots": [0, 0, 1]})
        assert refused.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as missing:
            _request(base + "/nowhere")
        assert missing.value.code == 404

        status, body = _request(base + "/new", {})
        assert status == 200 and json.loads(body)["deal"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
