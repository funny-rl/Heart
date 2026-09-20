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
import functools
import html
import importlib
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from numbers import Integral
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
    ClassicRules,
    ClassicState,
    observe_classic,
)
from heart.classic_render import render_classic_html
from heart.render import _visible_trick

RULE_TIERS = ("easy", "medium", "hard")
PLUGIN = re.compile(r"^(?P<module>[\w.]+):(?P<attr>\w+)(?:\((?P<argument>.*)\))?$")
MAX_REQUEST_BYTES = 16 * 1024
MAX_SEED = 2**32 - 1

# Interactive play takes one action at a time, so the eager dispatch cost that
# vanishes inside a batched scan would otherwise dominate a person's turn.
_STEP = jax.jit(ClassicEnv().step)
_OBSERVE = jax.jit(observe_classic)


class SeatPolicy(Protocol):
    """Choose one phase-appropriate action for `player` in `state`."""

    def __call__(self, state: ClassicState, player: int, key: jax.Array) -> int: ...


@functools.cache
def make_rule_seat_policy(difficulty: str) -> SeatPolicy:
    """Wrap the packaged rule tiers in the seat-policy calling convention.

    Cached per tier so seats sharing a difficulty share one compiled policy;
    three separate closures meant compiling the same tier three times, which a
    person waiting on the table feels as a pause on every early turn.
    """

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

    if not isinstance(slots, (list, tuple)) or any(
        isinstance(slot, bool) or not isinstance(slot, Integral) for slot in slots
    ):
        raise TypeError("pass slots must be integers")
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
        if isinstance(self.human_seat, bool) or not isinstance(
            self.human_seat, Integral
        ):
            raise TypeError("human seat must be an integer")
        self.human_seat = int(self.human_seat)
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
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        seed = int(seed)
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be in [0, {MAX_SEED}]")
        self.seed = seed
        self.key = jax.random.key(seed)
        self.key, reset_key = jax.random.split(self.key)
        self.state, _ = ClassicEnv().reset(reset_key)
        self.log = []

    @property
    def finished(self) -> bool:
        return int(self.state.phase) == TERMINAL

    @property
    def waiting_for_human(self) -> bool:
        return not self.finished and int(self.state.active_player) == self.human_seat

    @property
    def pending(self) -> bool:
        """A policy still owes an action before the person can act again."""

        return not self.finished and not self.waiting_for_human

    def advance_once(self) -> bool:
        """Play exactly one policy action, so a caller can pace the table."""

        if not self.pending:
            return False
        seat = int(self.state.active_player)
        next_key, action_key = jax.random.split(self.key)
        self._apply(seat, self.seats[seat](self.state, seat, action_key))
        self.key = next_key
        return True

    def advance(self) -> None:
        """Let the policies act until the person is on turn, or the match ends."""

        while self.advance_once():
            pass

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

    def legal_plays(self) -> list[int]:
        observation = _OBSERVE(self.state, self.human_seat)
        mask = np.asarray(jax.device_get(observation.play_action_mask))
        return [int(card) for card in np.flatnonzero(mask)]

    def snapshot(self, *, sweep: bool = False) -> dict:
        """One frame for the page.

        ``sweep`` clears a trick that has already been decided, which is what
        the person needs when they are about to lead the next one.
        """

        scores = np.asarray(jax.device_get(self.state.match_scores)).tolist()
        proxy = self.state.game._replace(active_player=self.state.active_player)
        trick, leader, settling = _visible_trick(proxy)
        payload = {
            "view": render_classic_html(
                self.state, self.human_seat, show_settled_trick=not sweep
            ),
            "phase": int(self.state.phase),
            "finished": self.finished,
            "your_turn": self.waiting_for_human,
            "seat": self.human_seat,
            "pending": self.pending,
            "active": int(self.state.active_player),
            "leader": int(leader),
            "settling": bool(settling) and not sweep,
            "trick": (
                []
                if (settling and sweep)
                else [
                    {
                        "seat": (int(leader) + offset) % NUM_PLAYERS,
                        "card": int(card),
                        "name": card_name(int(card)),
                    }
                    for offset, card in enumerate(np.asarray(trick))
                    if int(card) >= 0
                ]
            ),
            "scores": [int(value) for value in scores],
            "target": int(ClassicRules().target_score),
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
/* One screen: the table takes what is left after the controls. */
.wrap{height:100vh;max-width:1100px;margin:0 auto;padding:10px 12px;
  display:grid;grid-template-rows:auto minmax(0,1fr) auto;gap:10px}
h1{font-size:15px;margin:0;color:#9aa3b2;font-weight:600}
.stage{position:relative;overflow:hidden;min-height:0;border:1px solid #2a2f3a;
  border-radius:12px;background:#111}
iframe{position:absolute;top:0;left:50%;width:1080px;height:790px;border:0;
  transform-origin:top center;transform:translateX(-50%) scale(var(--k,1));
  opacity:1;transition:opacity .13s ease}
.bar{padding:10px 12px;border:1px solid #2a2f3a;border-radius:12px;background:#12151c}
.cards{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 4px;padding-top:12px}
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
@media(max-height:760px){button.card{width:50px;height:70px}}
@media(max-width:640px){button.card{width:46px;height:68px}
  button.card .corner{font-size:10px;line-height:9px}button.card .pip{font-size:22px}}
button.go{padding:9px 16px;border-radius:9px;border:1px solid #86efac;background:#123c2c;
  color:#86efac;font:600 15px system-ui;cursor:pointer}
button.go:disabled{opacity:.4;cursor:not-allowed}
.msg{color:#9aa3b2;margin:6px 0 0}
.row{display:flex;flex-wrap:wrap;align-items:center;gap:12px;justify-content:space-between}
.board{display:flex;gap:6px;flex-wrap:wrap;justify-content:center;margin:2px 0 6px}
.board b{display:flex;flex-direction:column;align-items:center;gap:1px;min-width:62px;padding:4px 8px;border-radius:9px;background:#1e293b;border:1px solid #334155;font-weight:600}
.board b.me{border-color:#38bdf8;background:#0c4a6e}
.board b.lead{border-color:#22c55e}
.board b.edge{border-color:#f59e0b}
.board i{font-style:normal;font-size:11px;opacity:.7;font-weight:500}
.board u{text-decoration:none;font-size:17px;font-variant-numeric:tabular-nums}
.turn{font:600 13px ui-monospace,monospace;color:#cbd5e1}
.turn b{color:#fcd34d}
select{height:34px;padding:0 8px;border-radius:8px;border:1px solid #39404e;
  background:#1b202a;color:#e8eaf0;font:600 13px system-ui;cursor:pointer}
.log{margin-top:7px;max-height:3.4em;overflow:hidden;font:12px/1.7 ui-monospace,monospace;color:#8b93a4;white-space:pre-wrap}
@media(max-width:640px){iframe{height:420px}}
</style></head><body><div class="wrap">
<h1>HEART — 사람 대전</h1>
<div class="stage" id="stage"><iframe id="view" title="현재 판"></iframe></div>
<div class="bar">
  <div class="row">
    <div id="turn" class="turn"></div>
    <label class="turn">진행 속도
      <select id="speed">
        <option value="1200">느리게</option>
        <option value="700" selected>보통</option>
        <option value="220">빠르게</option>
        <option value="0">즉시</option>
      </select>
    </label>
  </div>
  <div id="board" class="board"></div>
  <div id="prompt" class="msg">불러오는 중…</div>
  <div id="cards" class="cards"></div>
  <button id="submit" class="go" hidden>3장 넘기기</button>
  <button id="again" class="go" hidden>경기 초기화</button>
  <div id="log" class="log"></div>
</div></div><script>
let picked = [];
let timer = null;
let framed = false;
const view = document.getElementById('view');
const prompt = document.getElementById('prompt');
const cards = document.getElementById('cards');
const submit = document.getElementById('submit');
const again = document.getElementById('again');
const logBox = document.getElementById('log');
const turnBox = document.getElementById('turn');
const speed = document.getElementById('speed');
const stage = document.getElementById('stage');

function fit() {
  // Scale the fixed-size table document into whatever height is left over,
  // so the hand is always reachable without scrolling.
  const box = stage.getBoundingClientRect();
  if (!box.width || !box.height) return;
  stage.style.setProperty('--k', Math.min(box.width / 1080, box.height / 790));
}
window.addEventListener('resize', fit);

async function post(path, body) {
  const response = await fetch(path, {method: 'POST',
    headers: {'content-type': 'application/json'}, body: JSON.stringify(body || {})});
  if (!response.ok) { prompt.textContent = await response.text(); return null; }
  return response.json();
}
function drawBoard(s) {
  const board = document.getElementById('board');
  const scores = s.scores || [];
  if (!scores.length) { board.replaceChildren(); return; }
  const target = s.target || 100;
  const best = Math.min.apply(null, scores);
  const worst = Math.max.apply(null, scores);
  board.replaceChildren();
  scores.forEach((value, seat) => {
    const cell = document.createElement('b');
    cell.className = (seat === s.seat ? 'me' : '')
      + (value === best ? ' lead' : '')
      + (value !== worst || worst < target - 26 ? '' : ' edge');
    const who = document.createElement('i');
    who.textContent = seat === s.seat ? '나 (P' + seat + ')' : 'P' + seat;
    const num = document.createElement('u');
    num.textContent = value;
    cell.append(who, num);
    board.append(cell);
  });
  const note = document.createElement('b');
  note.append(Object.assign(document.createElement('i'), {textContent: '딜 ' + ((s.deal || 0) + 1)}));
  note.append(Object.assign(document.createElement('u'),
    {textContent: Math.max(target - worst, 0) + '점 남음'}));
  board.append(note);
}

function paint(markup) {
  // Swapping only the body keeps the parsed stylesheet, which is most of the
  // document; re-feeding srcdoc every frame makes the table stutter.
  const doc = framed ? view.contentDocument : null;
  if (!doc || !doc.body) { view.srcdoc = markup; framed = true; return; }
  const parsed = new DOMParser().parseFromString(markup, 'text/html');
  doc.body.replaceChildren(...parsed.body.childNodes);
}
async function pace(s) {
  if (timer !== null) { clearTimeout(timer); timer = null; }
  if (!s.pending) return;
  // One round trip per turn; the page then paces the frames locally.
  const reply = await post('/advance', {steps: 24});
  if (!reply || !reply.frames || !reply.frames.length) return;
  const base = Number(speed.value);
  let at = 0;
  const step = () => {
    const frame = reply.frames[at++];
    draw(frame);
    if (at < reply.frames.length) {
      // Hold a completed trick longer than an ordinary card.
      const wait = frame.settling ? Math.max(base, 240) * 2 : base;
      timer = setTimeout(step, wait);
    }
  };
  // The first reply also waits, so the seat after yours does not answer
  // the instant your card lands.
  timer = setTimeout(step, base);
}
function draw(s) {
  view.style.opacity = '.62';
  paint(s.view);
  setTimeout(() => { view.style.opacity = '1'; }, 40);
  turnBox.innerHTML = s.finished ? ''
    : '선턴 <b>P' + s.leader + '</b> · 차례 P' + s.active
      + (s.settling ? ' · 트릭 정리 중' : '');
  drawBoard(s);
  logBox.textContent = (s.log || []).join('\\n');
  fit();
  cards.replaceChildren();
  submit.hidden = true;
  again.hidden = false;
  again.textContent = s.finished ? '새 경기' : '경기 초기화';
  picked = [];
  if (s.finished) {
    if (timer !== null) { clearTimeout(timer); timer = null; }
    const won = (s.winners || []).includes(s.seat);
    prompt.textContent = won ? '이겼습니다.'
      : '경기 종료 — 승자 ' + (s.winners || []).map(p => 'P' + p).join(', ');
    return;
  }
  if (!s.your_turn) { prompt.textContent = '상대를 기다리는 중…'; return; }
  const passing = s.phase === 0;
  prompt.textContent = passing ? '넘길 카드 3장을 고르세요.' : '낼 카드를 고르세요.';
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
      button.onclick = () => act('/action', {card: item.card});
    }
    cards.append(button);
  }
}
async function act(path, body) {
  const s = await post(path, body);
  if (!s) return;
  draw(s);
  pace(s);
}
submit.onclick = () => act('/action', {slots: picked});
again.onclick = () => {
  const running = again.textContent === '경기 초기화';
  if (running && !confirm('진행 중인 경기를 버리고 새 판을 시작할까요?')) return;
  if (timer !== null) { clearTimeout(timer); timer = null; }
  act('/new', {});
};
fit();
fetch('/state').then(r => r.json()).then(s => { draw(s); pace(s); });
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
                    self._json(game.snapshot(sweep=game.waiting_for_human))
            else:
                self._send(b"not found", "text/plain; charset=utf-8", 404)

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("content-length") or 0)
                if not 0 <= length <= MAX_REQUEST_BYTES:
                    raise ValueError(
                        f"content-length must be in [0, {MAX_REQUEST_BYTES}]"
                    )
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise TypeError("request body must be a JSON object")
            except (json.JSONDecodeError, TypeError, ValueError):
                self.close_connection = True
                self._send(b"malformed json", "text/plain; charset=utf-8", 400)
                return
            with lock:
                try:
                    if self.path == "/new":
                        game.reset((game.seed + 1) & MAX_SEED)
                    elif self.path == "/advance":
                        steps = body.get("steps", 1)
                        if isinstance(steps, bool) or not isinstance(steps, Integral):
                            raise TypeError("steps must be an integer")
                        steps = int(steps)
                        if not 1 <= steps <= 64:
                            raise ValueError("steps must be 1..64")
                        frames = []
                        for _ in range(steps):
                            if not game.advance_once():
                                break
                            frames.append(game.snapshot())
                        if frames and frames[-1]["settling"] and game.waiting_for_human:
                            # Hold the decided trick, then sweep it before the
                            # person leads into an empty table.
                            frames.append(game.snapshot(sweep=True))
                        if "steps" in body:
                            self._json({"frames": frames})
                            return
                    elif self.path == "/action":
                        if "slots" in body:
                            game.play_human(pass_action_for(body["slots"]))
                        else:
                            game.play_human(body["card"])
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
                self._json(game.snapshot(sweep=game.waiting_for_human))

    return Handler


def warm_up(
    seats: dict[int, SeatPolicy],
    *,
    human_seat: int = 0,
    seed: int,
    actions: int = 24,
) -> None:
    """Compile the transition and every seat policy before anyone is waiting."""

    scratch = HumanGame(seats=seats, human_seat=human_seat, seed=seed)
    for _ in range(actions):
        if scratch.finished:
            break
        if scratch.pending:
            scratch.advance_once()
        elif int(scratch.state.phase) == PASS:
            scratch.play_human(pass_action_for([0, 1, 2]))
        else:
            scratch.play_human(scratch.legal_plays()[0])


def serve(game: HumanGame, host: str = "127.0.0.1", port: int = 8000):
    """Return a server bound to `host:port`; the caller runs and closes it."""

    return ThreadingHTTPServer((host, port), make_handler(game, threading.Lock()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m heart",
        description=__doc__.splitlines()[0],
    )
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
    seats = {seat: load_seat_policy(spec) for seat, spec in zip(others, specs)}
    warm_up(seats, human_seat=arguments.human_seat, seed=arguments.seed)
    game = HumanGame(
        seats=seats,
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
