"""Human-versus-policy play for `classic-v0` over a small local HTTP server.

The environment stays the authority: every human action is validated against the
same masks the compiled path uses, so a browser cannot introduce a move the
rules forbid.

Opponents are `SeatPolicy` callables.  Rule tiers ship with the package; a
learned policy is supplied through `--seat` with a `module:factory` reference,
which keeps training dependencies out of the environment distribution.
"""

from __future__ import annotations

import argparse
import html
import importlib
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from heart.agents import make_rule_pass_policy, make_rule_policy
from heart.cards import (
    HEARTS,
    NUM_PLAYERS,
    RANK_NAMES,
    SUIT_SYMBOLS,
    card_name,
)
from heart.classic import (
    PASS,
    PASS_COMBINATIONS,
    TERMINAL,
    ClassicEnv,
    ClassicState,
    observe_classic,
)
from heart.classic_render import render_classic_html

RULE_TIERS = ("easy", "medium", "hard")
PLUGIN = re.compile(r"^(?P<module>[\w.]+):(?P<attr>\w+)(?:\((?P<argument>.*)\))?$")

# Interactive play takes one action at a time, so the eager dispatch cost that
# vanishes inside a batched scan would otherwise dominate a person's turn.
_STEP = jax.jit(ClassicEnv().step)
_OBSERVE = jax.jit(observe_classic)


class SeatPolicy(Protocol):
    """Choose one phase-appropriate action for `player` in `state`."""

    def __call__(self, state: ClassicState, player: int, key: jax.Array) -> int: ...


def make_rule_seat_policy(difficulty: str) -> SeatPolicy:
    """Wrap the packaged rule tiers in the seat-policy calling convention."""

    if difficulty not in RULE_TIERS:
        raise ValueError(f"unknown rule tier: {difficulty}")
    pass_policy = make_rule_pass_policy(difficulty)
    play_policy = make_rule_policy(difficulty)

    @jax.jit
    def choose_pass(state: ClassicState, player: int, key: jax.Array) -> jax.Array:
        return pass_policy(observe_classic(state, player), key)

    @jax.jit
    def choose_play(state: ClassicState, player: int, key: jax.Array) -> jax.Array:
        return play_policy(observe_classic(state, player).game, key)

    def policy(state: ClassicState, player: int, key: jax.Array) -> int:
        chooser = choose_pass if int(state.phase) == PASS else choose_play
        return int(chooser(state, player, key))

    return policy


def load_seat_policy(spec: str) -> SeatPolicy:
    """Resolve `easy|medium|hard` or `package.module:factory(argument)`."""

    if spec in RULE_TIERS:
        return make_rule_seat_policy(spec)
    matched = PLUGIN.match(spec)
    if matched is None:
        raise ValueError(
            f"seat must be one of {RULE_TIERS} or 'package.module:factory(argument)',"
            f" got {spec!r}"
        )
    module = importlib.import_module(matched.group("module"))
    factory: Callable[..., SeatPolicy] = getattr(module, matched.group("attr"))
    argument = matched.group("argument")
    return factory() if argument is None else factory(argument)


def _card_payload(slot: int, card: int) -> dict:
    """Describe one card well enough for the page to draw a real face."""

    suit, rank = divmod(card, 13)
    return {
        "slot": slot,
        "card": card,
        "name": card_name(card),
        "rank": RANK_NAMES[rank],
        "suit": SUIT_SYMBOLS[suit],
        "red": suit in (1, HEARTS),
    }


def hand_cards(state: ClassicState, player: int) -> list[int]:
    """The player's remaining cards, ascending — the order pass slots index."""

    held = np.asarray(jax.device_get(state.game.hands))[player]
    return [int(card) for card in np.flatnonzero(held)]


def pass_action_for(slots: list[int]) -> int:
    """Map three hand slots to their `PASS_COMBINATIONS` row."""

    wanted = sorted(int(slot) for slot in slots)
    if len(set(wanted)) != 3:
        raise ValueError("passing takes exactly three distinct hand slots")
    combinations = np.asarray(PASS_COMBINATIONS)
    matches = np.flatnonzero((combinations == np.asarray(wanted)).all(axis=1))
    if matches.size != 1:
        raise ValueError(f"no pass combination for slots {wanted}")
    return int(matches[0])


@dataclass
class HumanGame:
    """One seat played by a person, the rest by policies."""

    seats: dict[int, SeatPolicy]
    human_seat: int = 0
    seed: int = 0
    state: ClassicState = field(init=False)
    key: jax.Array = field(init=False)
    log: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.human_seat < NUM_PLAYERS:
            raise ValueError(f"human seat must be 0..{NUM_PLAYERS - 1}")
        missing = [
            seat
            for seat in range(NUM_PLAYERS)
            if seat != self.human_seat and seat not in self.seats
        ]
        if missing:
            raise ValueError(f"no policy for seats {missing}")
        self.reset(self.seed)

    def reset(self, seed: int) -> None:
        self.seed = seed
        self.key = jax.random.key(seed)
        self.key, reset_key = jax.random.split(self.key)
        self.state, _ = ClassicEnv().reset(reset_key)
        self.log = []
        self.advance()

    @property
    def finished(self) -> bool:
        return int(self.state.phase) == TERMINAL

    @property
    def waiting_for_human(self) -> bool:
        return not self.finished and int(self.state.active_player) == self.human_seat

    def advance(self) -> None:
        """Let the policies act until the person is on turn, or the match ends."""

        while not self.finished and not self.waiting_for_human:
            seat = int(self.state.active_player)
            self.key, action_key = jax.random.split(self.key)
            action = int(self.seats[seat](self.state, seat, action_key))
            self._apply(seat, action)

    def _apply(self, seat: int, action: int) -> None:
        phase = int(self.state.phase)
        cards = hand_cards(self.state, seat)
        state, _, _, _, info = _STEP(self.state, jnp.asarray(action))
        if bool(info.invalid_action):
            raise ValueError(f"seat {seat} chose an illegal action: {action}")
        if phase == PASS:
            chosen = [cards[slot] for slot in np.asarray(PASS_COMBINATIONS)[action]]
            self.log.append(f"P{seat} 패스 {' '.join(card_name(c) for c in chosen)}")
        else:
            self.log.append(f"P{seat} {card_name(action)}")
        self.state = state

    def play_human(self, action: int) -> None:
        if self.finished:
            raise ValueError("the match is over")
        if not self.waiting_for_human:
            raise ValueError("it is not your turn")
        self._apply(self.human_seat, action)
        self.advance()

    def legal_plays(self) -> list[int]:
        observation = _OBSERVE(self.state, self.human_seat)
        mask = np.asarray(jax.device_get(observation.play_action_mask))
        return [int(card) for card in np.flatnonzero(mask)]

    def snapshot(self) -> dict:
        scores = np.asarray(jax.device_get(self.state.match_scores)).tolist()
        payload = {
            "view": render_classic_html(self.state, self.human_seat),
            "phase": int(self.state.phase),
            "finished": self.finished,
            "your_turn": self.waiting_for_human,
            "seat": self.human_seat,
            "scores": [int(value) for value in scores],
            "deal": int(self.state.deal_index),
            "log": self.log[-12:],
            "hand": [],
            "legal": [],
        }
        if self.waiting_for_human:
            cards = hand_cards(self.state, self.human_seat)
            payload["hand"] = [
                _card_payload(slot, card) for slot, card in enumerate(cards)
            ]
            if int(self.state.phase) != PASS:
                payload["legal"] = self.legal_plays()
        if self.finished:
            winners = np.asarray(jax.device_get(self.state.winner_mask)).tolist()
            payload["winners"] = [i for i, won in enumerate(winners) if won]
        return payload


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HEART — 사람 대전</title><style>
:root{color-scheme:dark}
body{margin:0;background:#0b0d12;color:#e8eaf0;font:15px/1.6 system-ui,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 12px}
iframe{width:100%;height:560px;border:1px solid #2a2f3a;border-radius:12px;background:#111}
.bar{margin-top:14px;padding:14px;border:1px solid #2a2f3a;border-radius:12px;background:#12151c}
.cards{display:flex;flex-wrap:wrap;gap:7px;margin:14px 0 4px;padding-top:12px}
button.card{position:relative;width:62px;height:88px;padding:0;border-radius:8px;
  border:1px solid #cbd5e1;background:linear-gradient(145deg,#fff,#e8edf4);
  color:#111827;box-shadow:0 5px 13px #0006;cursor:pointer;font:inherit;
  transition:transform .09s ease,box-shadow .09s ease}
button.card.red{color:#d91f45}
button.card .corner{position:absolute;top:5px;left:6px;text-align:left;
  font:800 13px/12px Georgia,serif}
button.card .pip{position:absolute;inset:0;display:grid;place-items:center;font-size:29px}
button.card:hover:not(:disabled){transform:translateY(-4px)}
button.card[aria-pressed=true]{transform:translateY(-11px);border:3px solid #fcd34d;
  box-shadow:0 10px 21px #0007,0 0 14px #fcd34d99}
button.card:disabled{opacity:.3;cursor:not-allowed;box-shadow:none;filter:grayscale(.6)}
@media(max-width:640px){button.card{width:46px;height:68px}
  button.card .corner{font-size:10px;line-height:9px}button.card .pip{font-size:22px}}
button.go{padding:9px 16px;border-radius:9px;border:1px solid #86efac;background:#123c2c;
  color:#86efac;font:600 15px system-ui;cursor:pointer}
button.go:disabled{opacity:.4;cursor:not-allowed}
.msg{color:#9aa3b2;margin:6px 0 0}
.log{margin-top:10px;font:12px ui-monospace,monospace;color:#8b93a4;white-space:pre-wrap}
@media(max-width:640px){iframe{height:420px}}
</style></head><body><div class="wrap">
<h1>HEART — 사람 대전</h1>
<iframe id="view" title="현재 판"></iframe>
<div class="bar">
  <div id="prompt" class="msg">불러오는 중…</div>
  <div id="cards" class="cards"></div>
  <button id="submit" class="go" hidden>3장 넘기기</button>
  <button id="again" class="go" hidden>새 경기</button>
  <div id="log" class="log"></div>
</div></div><script>
let picked = [];
const view = document.getElementById('view');
const prompt = document.getElementById('prompt');
const cards = document.getElementById('cards');
const submit = document.getElementById('submit');
const again = document.getElementById('again');
const logBox = document.getElementById('log');

async function post(path, body) {
  const response = await fetch(path, {method: 'POST',
    headers: {'content-type': 'application/json'}, body: JSON.stringify(body || {})});
  if (!response.ok) { prompt.textContent = await response.text(); return null; }
  return response.json();
}
function draw(s) {
  view.srcdoc = s.view;
  logBox.textContent = (s.log || []).join('\\n');
  cards.replaceChildren();
  submit.hidden = true;
  again.hidden = !s.finished;
  picked = [];
  if (s.finished) {
    const won = (s.winners || []).includes(s.seat);
    prompt.textContent = won ? '이겼습니다.'
      : '경기 종료 — 승자 ' + (s.winners || []).map(p => 'P' + p).join(', ');
    return;
  }
  if (!s.your_turn) { prompt.textContent = '상대를 기다리는 중…'; return; }
  const passing = s.phase === 0;
  prompt.textContent = passing ? '넘길 카드 3장을 고르세요.'
                               : '낼 카드를 고르세요.';
  submit.hidden = !passing;
  submit.disabled = true;
  for (const item of s.hand) {
    const button = document.createElement('button');
    button.className = item.red ? 'card red' : 'card';
    button.setAttribute('aria-label', item.name);
    const corner = document.createElement('span');
    corner.className = 'corner';
    corner.append(item.rank, document.createElement('br'), item.suit);
    const pip = document.createElement('span');
    pip.className = 'pip';
    pip.textContent = item.suit;
    button.append(corner, pip);
    if (passing) {
      button.setAttribute('aria-pressed', 'false');
      button.onclick = () => {
        const at = picked.indexOf(item.slot);
        if (at >= 0) picked.splice(at, 1);
        else if (picked.length < 3) picked.push(item.slot);
        button.setAttribute('aria-pressed', String(picked.includes(item.slot)));
        submit.disabled = picked.length !== 3;
      };
    } else {
      button.disabled = !s.legal.includes(item.card);
      button.onclick = async () => draw(await post('/action', {card: item.card}));
    }
    cards.append(button);
  }
}
submit.onclick = async () => { const s = await post('/action', {slots: picked});
  if (s) draw(s); };
again.onclick = async () => draw(await post('/new', {}));
fetch('/state').then(r => r.json()).then(draw);
</script></body></html>"""


def make_handler(game: HumanGame, lock: threading.Lock):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *arguments) -> None:
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: int = 200) -> None:
            self._send(json.dumps(payload).encode("utf-8"), "application/json", status)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/state":
                with lock:
                    self._json(game.snapshot())
            else:
                self._send(b"not found", "text/plain; charset=utf-8", 404)

        def do_POST(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(b"malformed json", "text/plain; charset=utf-8", 400)
                return
            with lock:
                try:
                    if self.path == "/new":
                        game.reset(game.seed + 1)
                    elif self.path == "/action":
                        if "slots" in body:
                            game.play_human(pass_action_for(body["slots"]))
                        else:
                            game.play_human(int(body["card"]))
                    else:
                        self._send(b"not found", "text/plain; charset=utf-8", 404)
                        return
                except (ValueError, KeyError, TypeError) as error:
                    self._send(
                        html.escape(str(error)).encode("utf-8"),
                        "text/plain; charset=utf-8",
                        400,
                    )
                    return
                self._json(game.snapshot())

    return Handler


def serve(game: HumanGame, host: str = "127.0.0.1", port: int = 8000):
    """Return a server bound to `host:port`; the caller runs and closes it."""

    return ThreadingHTTPServer((host, port), make_handler(game, threading.Lock()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seat",
        action="append",
        default=[],
        metavar="SPEC",
        help="opponent per remaining seat, in seat order; "
        "'easy|medium|hard' or 'package.module:factory(argument)'",
    )
    parser.add_argument("--human-seat", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args(argv)

    others = [seat for seat in range(NUM_PLAYERS) if seat != arguments.human_seat]
    specs = arguments.seat or ["medium"] * len(others)
    if len(specs) == 1:
        specs = specs * len(others)
    if len(specs) != len(others):
        raise SystemExit(f"--seat expects 1 or {len(others)} values, got {len(specs)}")
    game = HumanGame(
        seats={seat: load_seat_policy(spec) for seat, spec in zip(others, specs)},
        human_seat=arguments.human_seat,
        seed=arguments.seed,
    )
    server = serve(game, arguments.host, arguments.port)
    print(f"http://{arguments.host}:{server.server_address[1]}  (Ctrl+C 로 종료)")
    print("좌석: " + ", ".join(f"P{seat}={spec}" for seat, spec in zip(others, specs)))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
