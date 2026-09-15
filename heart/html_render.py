"""Dependency-free browser renderer for one HEART state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from heart.cards import (
    CARD_NAMES,
    CARD_RANKS,
    CARD_SUITS,
    HEARTS,
    NUM_PLAYERS,
    QUEEN_OF_SPADES,
    RANK_NAMES,
    SUIT_SYMBOLS,
)
from heart.types import State

_SEATS = ("bottom", "left", "top", "right")


def _face(card: int, legal: bool = False, small: bool = False) -> str:
    suit = int(CARD_SUITS[card])
    rank = RANK_NAMES[int(CARD_RANKS[card])]
    symbol = SUIT_SYMBOLS[suit]
    classes = ["card", "red" if suit in (1, HEARTS) else "black"]
    if legal:
        classes.append("legal")
    if small:
        classes.append("small")
    return (
        f'<div class="{" ".join(classes)}" aria-label="{CARD_NAMES[card]}">'
        f'<span class="corner">{rank}<br>{symbol}</span>'
        f'<span class="symbol">{symbol}</span></div>'
    )


def _captured(state: State, player: int) -> str:
    history = np.asarray(state.trick_history)
    winners = np.asarray(state.trick_winners)
    cards: list[int] = []
    for trick, winner in zip(history, winners):
        if int(winner) == player:
            cards.extend(int(card) for card in trick if int(card) >= 0)
    hearts = sum(int(CARD_SUITS[card]) == HEARTS for card in cards)
    queen = '<span class="q-token">Q♠</span>' if QUEEN_OF_SPADES in cards else ""
    return f'<span class="h-token">♥ {hearts}</span>{queen}'


def _seat(player: int, anchor: int) -> str:
    return _SEATS[(player - anchor) % NUM_PLAYERS]


def render_html(state: State, viewer: int | None = 0) -> str:
    """Return a full-observability HTML game-table snapshot.

    ``viewer`` only identifies the local player and rotates that seat to the
    bottom. Every player's hand remains visible. ``None`` selects an unanchored
    spectator view with player 0 at the bottom.
    """

    if viewer is not None and not 0 <= viewer < NUM_PLAYERS:
        raise ValueError(f"viewer must be in [0, {NUM_PLAYERS}) or None")

    anchor = 0 if viewer is None else viewer
    hands = np.asarray(state.hands)
    penalties = np.asarray(state.penalties)
    scores = np.asarray(state.scores)
    active = int(state.active_player)
    ended = bool(state.terminated)

    legal_mask = None
    if not ended:
        from heart.rules import legal_action_mask

        legal_mask = np.asarray(legal_action_mask(state))

    panels: list[str] = []
    hands_html: list[str] = []
    for player in range(NUM_PLAYERS):
        seat = _seat(player, anchor)
        classes = ["player", f"seat-{seat}"]
        if player == active and not ended:
            classes.append("active")
        if ended and bool(np.asarray(state.winner_mask)[player]):
            classes.append("winner")
        score = int(scores[player]) if ended else int(penalties[player])
        identity = " · 나" if player == viewer else ""
        turn = " · 차례" if player == active and not ended else ""
        panels.append(
            f'<section class="{" ".join(classes)}">'
            f'<strong>P{player}{identity}{turn}</strong><span>{score}점</span>'
            f'<small>{_captured(state, player)}</small></section>'
        )

        cards = [int(card) for card in np.flatnonzero(hands[player])]
        faces = "".join(
            _face(
                card,
                legal=(
                    player == active
                    and legal_mask is not None
                    and bool(legal_mask[card])
                ),
                small=seat != "bottom",
            )
            for card in cards
        )
        if not faces:
            faces = '<div class="empty">패 없음</div>'
        hands_html.append(
            f'<div class="hand hand-{seat}" aria-label="P{player} 손패">{faces}</div>'
        )

    trick_html: list[str] = []
    leader = int(state.leader)
    for offset, raw_card in enumerate(np.asarray(state.current_trick)):
        card = int(raw_card)
        if card < 0:
            continue
        player = (leader + offset) % NUM_PLAYERS
        seat = _seat(player, anchor)
        trick_html.append(
            f'<div class="trick-card trick-{seat}"><i>P{player}</i>{_face(card)}</div>'
        )

    if ended and int(state.moon_shooter) >= 0:
        status = f"P{int(state.moon_shooter)} 문샷 · 단독 우승"
    elif ended:
        winners = np.flatnonzero(np.asarray(state.winner_mask))
        status = "게임 종료 · " + ", ".join(f"P{int(p)}" for p in winners) + " 승리"
    else:
        status = f"P{active}의 차례"

    trick_number = min(int(state.trick_index) + 1, 13)
    heart_state = "하트 브로큰" if bool(state.hearts_broken) else "하트 잠김"

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HEART</title>
<style>
*{{box-sizing:border-box}}:root{{color-scheme:dark;font-family:Inter,Pretendard,system-ui,sans-serif}}
body{{margin:0;min-width:320px;min-height:100vh;display:grid;place-items:center;color:#f8fafc;background:radial-gradient(circle at 50% 0,#37235f,#101217 55%)}}
.game{{width:min(1080px,96vw);padding:16px}}.hud{{display:flex;align-items:center;justify-content:space-between;gap:14px;margin:0 18px 12px}}
.logo{{font-size:22px;font-weight:900;letter-spacing:.12em}}.logo b{{color:#fb7185}}.status{{font-weight:800}}.meta{{font-size:14px;color:#aab4c4}}
.table{{position:relative;min-height:680px;overflow:hidden;border:14px solid #68452d;border-radius:42%;background:radial-gradient(circle,rgba(255,255,255,.08),transparent 30%),repeating-linear-gradient(25deg,rgba(255,255,255,.012) 0 2px,transparent 2px 5px),#146446;box-shadow:inset 0 0 55px #062d24,0 24px 65px #0008}}
.player{{position:absolute;z-index:5;min-width:116px;padding:9px 12px;display:grid;gap:2px;text-align:center;border:1px solid #ffffff2b;border-radius:14px;background:#091c17d9;box-shadow:0 8px 20px #0005}}
.player strong{{font-size:14px}}.player>span{{font-size:13px;color:#d1fae5}}.player small{{display:flex;justify-content:center;gap:7px;min-height:18px}}.player.active{{border-color:#fcd34d;box-shadow:0 0 0 3px #fcd34d38}}.player.winner{{border-color:#86efac}}
.h-token{{color:#fda4af}}.q-token{{color:#e2e8f0}}.seat-top{{top:15px;left:50%;transform:translateX(-50%)}}.seat-bottom{{bottom:135px;left:50%;transform:translateX(-50%)}}.seat-left{{left:16px;top:43%;transform:translateY(-50%)}}.seat-right{{right:16px;top:43%;transform:translateY(-50%)}}
.hand{{position:absolute;z-index:4;display:flex;justify-content:center;pointer-events:none}}.hand-bottom{{left:50%;bottom:17px;transform:translateX(-50%);width:84%}}.hand-top{{left:50%;top:88px;transform:translateX(-50%);max-width:60%}}.hand-left{{left:86px;top:43%;transform:translateY(-50%);max-width:36%}}.hand-right{{right:86px;top:43%;transform:translateY(-50%);max-width:36%}}
.card{{position:relative;flex:0 0 62px;width:62px;height:88px;margin-left:-17px;border:1px solid #cbd5e1;border-radius:8px;background:linear-gradient(145deg,#fff,#e8edf4);color:#111827;box-shadow:0 5px 13px #0006}}.card:first-child{{margin-left:0}}.card.red{{color:#d91f45}}.card.legal{{transform:translateY(-9px);border:3px solid #fcd34d;box-shadow:0 8px 18px #0007,0 0 14px #fcd34d99}}.card.small{{flex-basis:39px;width:39px;height:56px;margin-left:-21px}}
.corner{{position:absolute;top:6px;left:7px;font:800 13px/12px Georgia,serif}}.symbol{{position:absolute;inset:0;display:grid;place-items:center;font-size:29px}}.small .corner{{top:4px;left:4px;font-size:9px;line-height:8px}}.small .symbol{{font-size:18px}}
.trick-zone{{position:absolute;z-index:2;left:50%;top:45%;width:260px;height:220px;transform:translate(-50%,-50%)}}.trick-card{{position:absolute}}.trick-card .card{{margin:0}}.trick-card i{{position:absolute;z-index:2;top:-8px;right:-8px;padding:2px 6px;border-radius:9px;background:#0f172a;font-size:10px;font-style:normal;font-weight:800}}.trick-top{{top:0;left:50%;transform:translateX(-50%)}}.trick-bottom{{bottom:0;left:50%;transform:translateX(-50%)}}.trick-left{{left:0;top:50%;transform:translateY(-50%)}}.trick-right{{right:0;top:50%;transform:translateY(-50%)}}.empty{{color:#ffffff91;font-size:13px}}
@media(max-width:700px){{.game{{width:100vw;padding:7px}}.hud{{margin:4px 8px 9px}}.meta{{display:none}}.table{{min-height:590px;border-width:8px;border-radius:31%}}.card{{flex-basis:46px;width:46px;height:68px;margin-left:-20px}}.corner{{font-size:10px;line-height:9px}}.symbol{{font-size:23px}}.seat-left{{left:5px}}.seat-right{{right:5px}}.trick-zone{{width:210px;height:188px}}}}
</style>
</head>
<body><div class="game">
<header class="hud"><div class="logo"><b>♥</b> HEART</div><div class="status">{status}</div><div class="meta">트릭 {trick_number}/13 · {heart_state}</div></header>
<main class="table" aria-label="HEART 카드 테이블">
{"".join(panels)}
{"".join(hands_html)}
<div class="trick-zone" aria-label="현재 트릭">{"".join(trick_html)}</div>
</main></div></body></html>"""


def save_html(state: State, path: str | Path, viewer: int | None = 0) -> Path:
    """Write a full-observability standalone snapshot and return its path."""

    output = Path(path).expanduser().resolve()
    output.write_text(render_html(state, viewer), encoding="utf-8")
    return output


def render_replay_html(
    states: Sequence[State], viewer: int | None = 0, fps: float = 2.0
) -> str:
    """Return a standalone, interactive full-deal replay document."""

    if not states:
        raise ValueError("states must contain at least one game state")
    if fps <= 0:
        raise ValueError("fps must be positive")

    frames = [render_html(state, viewer) for state in states]
    # Escaping '<' prevents a future renderer string from ending this script tag.
    frames_json = json.dumps(frames, ensure_ascii=False).replace("<", "\\u003c")
    interval = round(1000.0 / fps)
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HEART Replay</title><style>
*{{box-sizing:border-box}}body{{margin:0;min-width:320px;color:#f8fafc;background:#0b0d12;font-family:Inter,Pretendard,system-ui,sans-serif}}
.replay{{height:100vh;display:grid;grid-template-rows:minmax(0,1fr) auto}}iframe{{width:100%;height:100%;border:0;opacity:1;transition:opacity .1s ease}}
.controls{{display:grid;grid-template-columns:auto auto minmax(120px,1fr) auto auto;gap:12px;align-items:center;padding:12px 18px;background:#171a22;border-top:1px solid #ffffff20;box-shadow:0 -8px 30px #0007}}
button,select{{height:38px;border:1px solid #ffffff2b;border-radius:9px;color:#f8fafc;background:#272c38;font:inherit;font-weight:750;cursor:pointer}}button{{min-width:44px}}button:hover{{background:#343b4a}}input{{width:100%;accent-color:#fb7185}}output{{min-width:72px;text-align:center;font-variant-numeric:tabular-nums;color:#cbd5e1}}
@media(max-width:600px){{.controls{{grid-template-columns:auto minmax(100px,1fr) auto;padding:9px;gap:7px}}#prev,#next{{display:none}}}}
</style></head><body><main class="replay">
<iframe id="stage" title="HEART 경기 리플레이"></iframe>
<nav class="controls" aria-label="리플레이 조작">
<button id="prev" aria-label="이전 프레임">◀</button><button id="play" aria-label="재생 또는 일시정지">▶</button>
<input id="timeline" type="range" min="0" max="{len(frames) - 1}" value="0" step="1" aria-label="경기 진행 위치">
<output id="counter">1 / {len(frames)}</output>
<select id="speed" aria-label="재생 속도"><option value="2">0.5×</option><option value="1" selected>1×</option><option value="0.5">2×</option><option value="0.25">4×</option></select>
<button id="next" aria-label="다음 프레임">▶|</button></nav></main>
<script>
const frames={frames_json};const baseInterval={interval};let index=0;let timer=null;
const stage=document.querySelector('#stage'),timeline=document.querySelector('#timeline'),counter=document.querySelector('#counter'),play=document.querySelector('#play'),speed=document.querySelector('#speed');
function show(next){{index=Math.max(0,Math.min(frames.length-1,next));stage.style.opacity='.35';setTimeout(()=>{{stage.srcdoc=frames[index];stage.style.opacity='1'}},70);timeline.value=index;counter.value=`${{index+1}} / ${{frames.length}}`;}}
function stop(){{if(timer!==null)clearInterval(timer);timer=null;play.textContent='▶';}}
function start(){{if(index===frames.length-1)show(0);stop();play.textContent='Ⅱ';timer=setInterval(()=>{{if(index===frames.length-1)stop();else show(index+1)}},baseInterval*Number(speed.value));}}
play.onclick=()=>timer===null?start():stop();document.querySelector('#prev').onclick=()=>{{stop();show(index-1)}};document.querySelector('#next').onclick=()=>{{stop();show(index+1)}};timeline.oninput=()=>{{stop();show(Number(timeline.value))}};speed.onchange=()=>{{if(timer!==null)start()}};
document.onkeydown=e=>{{if(e.key===' '){{e.preventDefault();play.click()}}else if(e.key==='ArrowLeft')document.querySelector('#prev').click();else if(e.key==='ArrowRight')document.querySelector('#next').click();}};show(0);
</script></body></html>"""


def save_replay_html(
    states: Sequence[State],
    path: str | Path,
    viewer: int | None = 0,
    fps: float = 2.0,
) -> Path:
    """Write an interactive replay as one portable HTML file."""

    output = Path(path).expanduser().resolve()
    output.write_text(render_replay_html(states, viewer, fps), encoding="utf-8")
    return output
