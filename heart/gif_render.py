"""Optional Pillow renderer for compact animated match previews."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from heart.cards import (
    CARD_RANKS,
    CARD_SUITS,
    HEARTS,
    NUM_PLAYERS,
    RANK_NAMES,
    SUIT_SYMBOLS,
)
from heart.rules import legal_action_mask
from heart.types import State

_SEATS = ("bottom", "left", "top", "right")


def _pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "GIF rendering requires Pillow; install heart-marl[render]"
        ) from exc
    return Image, ImageDraw, ImageFont


def _font(image_font: Any, size: int, *, bold: bool = False) -> Any:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return image_font.truetype(name, size)
    except OSError:  # pragma: no cover - platform font fallback
        return image_font.load_default()


def _center(draw: Any, xy: tuple[int, int], text: str, font: Any, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
        text,
        font=font,
        fill=fill,
    )


def _card(
    draw: Any,
    card: int,
    xy: tuple[int, int],
    size: tuple[int, int],
    fonts: tuple[Any, Any],
    *,
    legal: bool = False,
) -> None:
    x, y = xy
    width, height = size
    suit = int(CARD_SUITS[card])
    rank = RANK_NAMES[int(CARD_RANKS[card])]
    symbol = SUIT_SYMBOLS[suit]
    color = "#d91f45" if suit in (1, HEARTS) else "#111827"
    outline = "#fbbf24" if legal else "#cbd5e1"
    stroke = 4 if legal else 1
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=max(4, width // 8),
        fill="#f8fafc",
        outline=outline,
        width=stroke,
    )
    draw.text((x + 4, y + 2), rank, font=fonts[0], fill=color)
    _center(draw, (x + width // 2, y + height // 2 + 3), symbol, fonts[1], color)


def _seat(player: int, viewer: int | None) -> str:
    anchor = 0 if viewer is None else viewer
    return _SEATS[(player - anchor) % NUM_PLAYERS]


def render_gif_frame(
    state: State,
    viewer: int | None = 0,
    size: tuple[int, int] = (960, 540),
) -> Any:
    """Render one full-observability state as a Pillow RGB image."""

    if viewer is not None and not 0 <= viewer < NUM_PLAYERS:
        raise ValueError("viewer must be in [0, 4) or None")
    if size[0] < 640 or size[1] < 360:
        raise ValueError("GIF frame size must be at least 640x360")

    image, image_draw, image_font = _pillow()
    canvas = image.new("RGB", size, "#11131a")
    draw = image_draw.Draw(canvas)
    width, height = size
    scale = min(width / 960, height / 540)

    def box(values: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return tuple(round(value * scale) for value in values)

    title_font = _font(image_font, round(24 * scale), bold=True)
    ui_font = _font(image_font, round(14 * scale), bold=True)
    small_font = _font(image_font, round(11 * scale), bold=True)
    card_font = _font(image_font, round(12 * scale), bold=True)
    suit_font = _font(image_font, round(24 * scale))
    tiny_card_font = _font(image_font, round(9 * scale), bold=True)
    tiny_suit_font = _font(image_font, round(17 * scale))

    draw.rounded_rectangle(box((18, 43, 942, 530)), radius=55, fill="#68452d")
    draw.rounded_rectangle(box((31, 56, 929, 517)), radius=48, fill="#125c42")
    draw.ellipse(
        box((210, 102, 750, 474)), outline="#ffffff18", width=max(1, round(scale))
    )
    draw.text(
        (round(24 * scale), round(10 * scale)),
        "♥ HEART",
        font=title_font,
        fill="#f8fafc",
    )
    status = (
        "FINAL"
        if bool(state.terminated)
        else f"TRICK {min(int(state.trick_index) + 1, 13)}/13  ·  ACTION {int(state.num_cards_played)}/52"
    )
    draw.text(
        (round(730 * scale), round(15 * scale)), status, font=small_font, fill="#cbd5e1"
    )

    hands = np.asarray(state.hands)
    active = int(state.active_player)
    penalties = np.asarray(state.penalties)
    scores = np.asarray(state.scores)
    legal = None if bool(state.terminated) else np.asarray(legal_action_mask(state))
    panel_centers = {
        "bottom": (480, 430),
        "top": (480, 72),
        "left": (92, 176),
        "right": (868, 176),
    }
    hand_origins = {
        "bottom": (290, 455),
        "top": (328, 91),
        "left": (57, 202),
        "right": (869, 202),
    }

    for player in range(NUM_PLAYERS):
        seat = _seat(player, viewer)
        px, py = panel_centers[seat]
        current_score = int(
            scores[player] if bool(state.terminated) else penalties[player]
        )
        identity = " · YOU" if player == viewer else ""
        turn = " · TURN" if player == active and not bool(state.terminated) else ""
        label = f"P{player}{identity}{turn}  {current_score}pt"
        label_box = draw.textbbox((0, 0), label, font=small_font)
        pad = round(7 * scale)
        lw = label_box[2] - label_box[0] + 2 * pad
        lh = label_box[3] - label_box[1] + 2 * pad
        fill = "#594915" if turn else "#09231bdc"
        outline = "#fbbf24" if turn else "#ffffff35"
        draw.rounded_rectangle(
            (
                round(px * scale - lw / 2),
                round(py * scale - lh / 2),
                round(px * scale + lw / 2),
                round(py * scale + lh / 2),
            ),
            radius=round(8 * scale),
            fill=fill,
            outline=outline,
            width=max(1, round(2 * scale)),
        )
        _center(
            draw, (round(px * scale), round(py * scale)), label, small_font, "#f8fafc"
        )

        cards = [int(card) for card in np.flatnonzero(hands[player])]
        ox, oy = hand_origins[seat]
        if seat in ("bottom", "top"):
            card_size = (round(42 * scale), round(59 * scale))
            gap = round((30 if seat == "bottom" else 25) * scale)
            total = card_size[0] + gap * max(len(cards) - 1, 0)
            start_x = round(width / 2 - total / 2)
            for index, card in enumerate(cards):
                _card(
                    draw,
                    card,
                    (start_x + index * gap, round(oy * scale)),
                    card_size,
                    (card_font, suit_font),
                    legal=player == active and legal is not None and bool(legal[card]),
                )
        else:
            card_size = (round(34 * scale), round(48 * scale))
            gap = round(17 * scale)
            total = card_size[1] + gap * max(len(cards) - 1, 0)
            start_y = round(height / 2 - total / 2 + 20 * scale)
            for index, card in enumerate(cards):
                _card(
                    draw,
                    card,
                    (round(ox * scale), start_y + index * gap),
                    card_size,
                    (tiny_card_font, tiny_suit_font),
                    legal=player == active and legal is not None and bool(legal[card]),
                )

    trick_positions = {
        "bottom": (459, 341),
        "top": (459, 221),
        "left": (378, 281),
        "right": (540, 281),
    }
    leader = int(state.leader)
    for offset, raw_card in enumerate(np.asarray(state.current_trick)):
        card = int(raw_card)
        if card < 0:
            continue
        player = (leader + offset) % NUM_PLAYERS
        tx, ty = trick_positions[_seat(player, viewer)]
        _card(
            draw,
            card,
            (round(tx * scale), round(ty * scale)),
            (round(43 * scale), round(60 * scale)),
            (card_font, suit_font),
        )

    if bool(state.terminated):
        shooter = int(state.moon_shooter)
        result = f"P{shooter} SHOT THE MOON" if shooter >= 0 else "DEAL COMPLETE"
        draw.rounded_rectangle(
            box((320, 276, 640, 330)),
            radius=14,
            fill="#111827e8",
            outline="#86efac",
            width=2,
        )
        _center(
            draw, (round(480 * scale), round(303 * scale)), result, ui_font, "#86efac"
        )

    return canvas


def save_gif(
    states: Sequence[State],
    path: str | Path,
    viewer: int | None = 0,
    duration_ms: int = 95,
    size: tuple[int, int] = (960, 540),
) -> Path:
    """Write state snapshots as a looping optimized GIF."""

    if not states:
        raise ValueError("states must contain at least one game state")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    frames = [render_gif_frame(state, viewer, size) for state in states]
    output = Path(path).expanduser().resolve()
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return output
