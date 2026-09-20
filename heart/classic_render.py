"""Seat-view and omniscient renderers for classic-v0 match boundaries."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from jax import device_get

from heart.cards import CARD_NAMES, NUM_PLAYERS
from heart.classic import (
    PASS,
    PASS_NAMES,
    PLAY,
    ClassicState,
)
from heart.gif_render import _center, _font, _pillow, render_gif_frame
from heart.html_render import render_html, render_replay_html
from heart.render import _validate_viewer

_DIRECTION_LABELS = {
    "left": "왼쪽",
    "right": "오른쪽",
    "across": "맞은편",
    "hold": "홀드",
}


def _phase_text(state: ClassicState) -> str:
    if bool(state.terminated):
        winners = np.flatnonzero(np.asarray(state.winner_mask))
        return "경기 종료 · " + ", ".join(f"P{int(p)}" for p in winners) + " 승리"
    direction = PASS_NAMES[int(state.pass_direction)]
    if int(state.phase) == PASS:
        return (
            f"딜 {int(state.deal_index) + 1} · "
            f"{_DIRECTION_LABELS[direction]} 패싱 · "
            f"P{int(state.active_player)} 선택"
        )
    return f"딜 {int(state.deal_index) + 1} · 플레이 · P{int(state.active_player)} 차례"


def _deal_summary_html(state: ClassicState) -> str:
    if not bool(state.deal_boundary):
        return ""
    completed = (
        int(state.deal_index)
        if not bool(state.terminated)
        else int(state.deal_index) + 1
    )
    cells = "".join(
        (
            f"<span><b>P{player}</b> "
            f"{int(state.last_deal_scores[player])}점 "
            f"<em>{float(state.last_deal_rewards[player]):+.2f}</em></span>"
        )
        for player in range(NUM_PLAYERS)
    )
    moon = (
        f'<strong class="moon">P{int(state.last_deal_moon_shooter)} 문샷</strong>'
        if int(state.last_deal_moon_shooter) >= 0
        else ""
    )
    next_text = (
        "최종 경기 결과"
        if bool(state.terminated)
        else (
            "다음: 홀드 · 바로 플레이 (패싱 없음)"
            if int(state.phase) == PLAY
            else f"다음: {_DIRECTION_LABELS[PASS_NAMES[int(state.pass_direction)]]} 패싱"
        )
    )
    cards = " · ".join(
        CARD_NAMES[int(card)] for card in state.last_trick_cards if int(card) >= 0
    )
    cards = cards or "기록 없음"
    return (
        f'<aside class="deal-summary"><strong>딜 {completed} 정산</strong>'
        f"<small>마지막 트릭 · P{int(state.last_trick_winner)} 획득 · {cards}</small>"
        f"<div>{cells}</div>{moon}<small>{next_text}</small></aside>"
    )


def _pass_flow_html(state: ClassicState, viewer: int | None = None) -> str:
    if int(state.phase) != PASS or bool(state.terminated) or bool(state.deal_boundary):
        return ""
    selected = np.asarray(state.pass_cards)
    seats = []
    for player in range(NUM_PLAYERS):
        cards = [CARD_NAMES[int(card)] for card in selected[player] if int(card) >= 0]
        if cards and viewer is not None and player != viewer:
            value = "선택 완료"
        else:
            value = " · ".join(cards) if cards else "선택 대기"
        state_class = " done" if cards else ""
        seats.append(
            f'<span class="pass-seat{state_class}"><b>P{player}</b>{value}</span>'
        )
    direction = _DIRECTION_LABELS[PASS_NAMES[int(state.pass_direction)]]
    return (
        f'<section class="pass-flow"><strong>3장 {direction} 전달</strong>'
        f"<div>{''.join(seats)}</div></section>"
    )


def render_classic_html(
    state: ClassicState,
    viewer: int | None = 0,
    *,
    show_settled_trick: bool = True,
) -> str:
    """Return a standalone classic match snapshot with inter-deal context."""

    state = device_get(state)
    proxy = state.game._replace(active_player=state.active_player)
    rendered = render_html(
        proxy,
        viewer,
        reward_override=np.asarray(state.last_deal_rewards),
        winner_override=np.asarray(state.winner_mask),
        show_settled_trick=show_settled_trick,
    ).replace("누적 벌점", "이번 딜")
    if int(state.phase) == PASS:
        rendered = rendered.replace(" legal", "")
    rendered = re.sub(
        r'<div class="status">.*?</div>',
        f'<div class="status">{_phase_text(state)}</div>',
        rendered,
        count=1,
    )
    score_cells = "".join(
        f'<span class="{"winner" if bool(state.winner_mask[player]) else ""}">'
        f"<b>P{player}</b>{int(state.match_scores[player])}</span>"
        for player in range(NUM_PLAYERS)
    )
    match_hud = (
        f'<section class="match-score" aria-label="누적 경기 점수">'
        f"<strong>100점 경기</strong>{score_cells}</section>"
    )
    extra_css = """
.match-score{display:flex;align-items:center;justify-content:center;gap:8px;margin:0 auto 10px}
.match-score>strong{margin-right:5px;color:#fda4af}.match-score span{display:flex;gap:6px;min-width:66px;justify-content:center;padding:5px 9px;border:1px solid #ffffff2b;border-radius:10px;background:#171a22;font-variant-numeric:tabular-nums}.match-score span b{color:#94a3b8}
.match-score span.winner{border-color:#86efac;background:#123c2c}.match-score span.winner b{color:#86efac}
.deal-summary{position:absolute;z-index:20;left:50%;top:45%;transform:translate(-50%,-50%);width:min(520px,78%);padding:16px;border:1px solid #86efac;border-radius:18px;text-align:center;background:#071713f2;box-shadow:0 15px 45px #000a}
.deal-summary>div,.pass-flow>div{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin:12px 0}.deal-summary span,.pass-seat{display:grid;gap:3px;padding:8px 5px;border-radius:9px;background:#ffffff0d;font-size:12px}.deal-summary em{color:#93c5fd;font-style:normal}.deal-summary small{color:#cbd5e1}.moon{display:block;margin:5px;color:#fcd34d}
.pass-flow{position:absolute;z-index:20;left:50%;top:45%;transform:translate(-50%,-50%);width:min(520px,78%);padding:16px;border:1px solid #fcd34d;border-radius:18px;text-align:center;background:#071713f2;box-shadow:0 15px 45px #000a}.pass-seat{color:#94a3b8}.pass-seat.done{color:#f8fafc;background:#ffffff19}.pass-seat b{color:#fcd34d}
"""
    rendered = rendered.replace("</style>", extra_css + "</style>", 1)
    rendered = rendered.replace(
        '<main class="table"',
        match_hud + '<main class="table"',
        1,
    )
    overlay = _deal_summary_html(state) + _pass_flow_html(state, viewer)
    rendered = rendered.replace(
        "</main></div></body>", overlay + "</main></div></body>"
    )
    return rendered


def save_classic_html(
    state: ClassicState,
    path: str | Path,
    viewer: int | None = 0,
) -> Path:
    output = Path(path).expanduser().resolve()
    output.write_text(render_classic_html(state, viewer), encoding="utf-8")
    return output


def render_classic_replay_html(
    states: Sequence[ClassicState],
    viewer: int | None = 0,
    fps: float = 4.0,
) -> str:
    """Return one portable interactive replay for classic event frames."""

    return render_replay_html(
        states,
        viewer,
        fps,
        frame_renderer=render_classic_html,
    )


def save_classic_replay_html(
    states: Sequence[ClassicState],
    path: str | Path,
    viewer: int | None = 0,
    fps: float = 4.0,
) -> Path:
    """Write a classic match replay with timeline and speed controls."""

    output = Path(path).expanduser().resolve()
    output.write_text(
        render_classic_replay_html(states, viewer, fps),
        encoding="utf-8",
    )
    return output


def render_classic_ansi(state: ClassicState, viewer: int | None = 0) -> str:
    """Render a classic match snapshot as readable text from a seat's view.

    ``viewer`` sees only its own hand and pass selection; ``None`` reveals all.
    """

    state = device_get(state)
    viewer = _validate_viewer(viewer)
    lines = [
        _phase_text(state),
        "Match scores: "
        + "  ".join(
            f"P{player}={int(state.match_scores[player])}"
            for player in range(NUM_PLAYERS)
        ),
    ]
    if bool(state.deal_boundary):
        lines.append(
            "Last deal: "
            + "  ".join(
                (
                    f"P{player}={int(state.last_deal_scores[player])}"
                    f"({float(state.last_deal_rewards[player]):+.2f})"
                )
                for player in range(NUM_PLAYERS)
            )
        )
    if int(state.phase) == PASS and not bool(state.deal_boundary):
        choices = []
        for player in range(NUM_PLAYERS):
            chosen = [
                CARD_NAMES[int(card)]
                for card in state.pass_cards[player]
                if int(card) >= 0
            ]
            if chosen and viewer is not None and player != viewer:
                chosen = ["??"] * len(chosen)
            choices.append(f"P{player}=" + " ".join(chosen))
        lines.append("Pass choices: " + "  ".join(choices))
    hands = np.asarray(state.game.hands)
    for player in range(NUM_PLAYERS):
        if viewer is None or player == viewer:
            cards = " ".join(
                CARD_NAMES[int(card)] for card in np.flatnonzero(hands[player])
            )
        else:
            cards = " ".join(["??"] * int(hands[player].sum())) or "-"
        marker = (
            "→"
            if not bool(state.terminated) and player == int(state.active_player)
            else " "
        )
        identity = " (YOU)" if player == viewer else ""
        lines.append(f"{marker} P{player}{identity}: {cards}")
    return "\n".join(lines)


def render_classic_gif_frame(
    state: ClassicState,
    viewer: int | None = 0,
    size: tuple[int, int] = (960, 540),
) -> Any:
    """Render one classic match frame, including pass and deal overlays."""

    viewer = _validate_viewer(viewer)
    state = device_get(state)
    proxy = state.game._replace(active_player=state.active_player)
    canvas = render_gif_frame(
        proxy,
        viewer,
        size,
        show_legal=int(state.phase) == PLAY,
        reward_override=np.asarray(state.last_deal_rewards),
        winner_override=np.asarray(state.winner_mask),
    )
    _, image_draw, image_font = _pillow()
    draw = image_draw.Draw(canvas)
    width, height = size
    scale = min(width / 960, height / 540)
    ox = (width - 960 * scale) / 2
    oy = (height - 540 * scale) / 2

    def sx(value: float) -> int:
        return round(ox + value * scale)

    def sy(value: float) -> int:
        return round(oy + value * scale)

    font = _font(image_font, round(12 * scale), bold=True)
    small = _font(image_font, round(10 * scale), bold=True)

    scoreboard = "  ".join(
        f"P{p} {int(state.match_scores[p])}" for p in range(NUM_PLAYERS)
    )
    draw.rounded_rectangle(
        (sx(255), sy(7), sx(705), sy(38)),
        radius=round(10 * scale),
        fill="#171a22ed",
        outline="#ffffff35",
    )
    _center(draw, (sx(480), sy(23)), f"MATCH TO 100  ·  {scoreboard}", small, "#f8fafc")

    if (
        int(state.phase) == PASS
        and not bool(state.terminated)
        and not bool(state.deal_boundary)
    ):
        direction = PASS_NAMES[int(state.pass_direction)].upper()
        chosen = int(np.all(np.asarray(state.pass_cards) >= 0, axis=1).sum())
        text = (
            f"PASS {direction}  ·  P{int(state.active_player)} SELECTS"
            f"  ·  {chosen}/4 READY"
        )
        draw.rounded_rectangle(
            (sx(275), sy(264), sx(685), sy(318)),
            radius=round(14 * scale),
            fill="#111827ed",
            outline="#fcd34d",
            width=max(1, round(2 * scale)),
        )
        _center(draw, (sx(480), sy(291)), text, font, "#fcd34d")
        selected = np.asarray(state.pass_cards)
        choices = []
        for player in range(NUM_PLAYERS):
            cards = [
                CARD_NAMES[int(card)] for card in selected[player] if int(card) >= 0
            ]
            if not cards:
                continue
            if viewer is not None and player != viewer:
                cards = ["??"] * len(cards)
            choices.append(f"P{player} " + "/".join(cards))
        choice_text = "  ".join(choices)
        if choice_text:
            _center(draw, (sx(480), sy(311)), choice_text, small, "#f8fafc")

    if bool(state.deal_boundary):
        deal_number = (
            int(state.deal_index)
            if not bool(state.terminated)
            else int(state.deal_index) + 1
        )
        scores = "  ".join(
            (
                f"P{p} {int(state.last_deal_scores[p])}"
                f" ({float(state.last_deal_rewards[p]):+.2f})"
            )
            for p in range(NUM_PLAYERS)
        )
        draw.rounded_rectangle(
            (sx(220), sy(325), sx(740), sy(380)),
            radius=round(14 * scale),
            fill="#071713f2",
            outline="#86efac",
            width=max(1, round(2 * scale)),
        )
        _center(
            draw, (sx(480), sy(342)), f"DEAL {deal_number} COMPLETE", font, "#86efac"
        )
        _center(draw, (sx(480), sy(363)), scores, small, "#cbd5e1")
        cards = "  ".join(
            CARD_NAMES[int(card)] for card in state.last_trick_cards if int(card) >= 0
        )
        _center(draw, (sx(480), sy(386)), cards, small, "#f8fafc")
    if bool(state.terminated):
        winners = ", ".join(
            f"P{player}" for player in np.flatnonzero(np.asarray(state.winner_mask))
        )
        _center(
            draw,
            (sx(480), sy(410)),
            f"MATCH COMPLETE · WINNER {winners}",
            font,
            "#86efac",
        )
    return canvas


def save_classic_gif(
    states: Sequence[ClassicState],
    path: str | Path,
    viewer: int | None = 0,
    duration_ms: int = 120,
    size: tuple[int, int] = (960, 540),
) -> Path:
    if not states:
        raise ValueError("states must contain at least one classic state")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".gif":
        raise ValueError("GIF output path must use the .gif suffix")
    frames = [render_classic_gif_frame(state, viewer, size) for state in states]
    durations = [
        max(duration_ms, 1200) if bool(state.deal_boundary) else duration_ms
        for state in states
    ]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
        format="GIF",
    )
    return output
