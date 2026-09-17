"""Optional Pillow renderer for compact animated match previews."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from jax import device_get

from heart.cards import (
    HEARTS,
    NUM_CARDS,
    NUM_PLAYERS,
    RANK_NAMES,
    SUIT_SYMBOLS,
)
from heart.render import _terminal_rewards, _visible_trick
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
    suit, rank_index = divmod(card, 13)
    rank = RANK_NAMES[rank_index]
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


def _card_back(draw: Any, xy: tuple[int, int], size: tuple[int, int]) -> None:
    """Draw a face-down card, used for every hand that is not the viewer's."""

    x, y = xy
    width, height = size
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=max(4, width // 8),
        fill="#1b3a5c",
        outline="#7f93ad",
        width=1,
    )
    inset = max(3, width // 9)
    draw.rounded_rectangle(
        (x + inset, y + inset, x + width - inset, y + height - inset),
        radius=max(2, width // 12),
        outline="#4d7bb0",
        width=max(1, width // 20),
    )


def _seat(player: int, viewer: int | None) -> str:
    anchor = 0 if viewer is None else viewer
    return _SEATS[(player - anchor) % NUM_PLAYERS]


def render_gif_frame(
    state: State,
    viewer: int | None = 0,
    size: tuple[int, int] = (960, 540),
    *,
    show_legal: bool = True,
    reward_override: np.ndarray | None = None,
    winner_override: np.ndarray | None = None,
) -> Any:
    """Render one state as a Pillow RGB image from a seat's point of view.

    ``viewer`` is the only hand drawn face up; the others show face-down cards
    in their true counts. ``None`` reveals every hand.
    """

    if viewer is not None and not 0 <= viewer < NUM_PLAYERS:
        raise ValueError("viewer must be in [0, 4) or None")
    if size[0] < 640 or size[1] < 360:
        raise ValueError("GIF frame size must be at least 640x360")

    state, host_legal = device_get((state, legal_action_mask(state)))
    image, image_draw, image_font = _pillow()
    canvas = image.new("RGB", size, "#11131a")
    draw = image_draw.Draw(canvas)
    width, height = size
    scale = min(width / 960, height / 540)
    offset_x = (width - 960 * scale) / 2
    offset_y = (height - 540 * scale) / 2

    def sx(value: float) -> int:
        return round(offset_x + value * scale)

    def sy(value: float) -> int:
        return round(offset_y + value * scale)

    def box(values: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = values
        return sx(x1), sy(y1), sx(x2), sy(y2)

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
        (sx(24), sy(10)),
        "♥ HEART",
        font=title_font,
        fill="#f8fafc",
    )
    status = (
        "FINAL"
        if bool(state.terminated)
        else f"TRICK {min(int(state.trick_index) + 1, 13)}/13  ·  ACTION {int(state.num_cards_played)}/52"
    )
    draw.text((sx(730), sy(15)), status, font=small_font, fill="#cbd5e1")

    hands = np.asarray(state.hands)
    active = int(state.active_player)
    penalties = np.asarray(state.penalties)
    scores = np.asarray(state.scores)
    ended = bool(state.terminated)
    rewards = (
        np.asarray(reward_override)
        if ended and reward_override is not None
        else _terminal_rewards(state)
        if ended
        else None
    )
    winners = (
        np.asarray(winner_override)
        if winner_override is not None
        else np.asarray(state.winner_mask)
    )
    legal = None if ended or not show_legal else np.asarray(host_legal)
    panel_columns = {"bottom": 480, "top": 480, "left": 92, "right": 868}
    hand_origins = {
        "bottom": (290, 455),
        "top": (328, 91),
        "left": (57, 202),
        "right": (869, 202),
    }
    # A seat badge is placed from the cards it labels rather than at a fixed
    # point, so a full 13-card hand cannot end up underneath it.  Side columns
    # use their widest extent, which keeps the badge still while cards are
    # played out.
    badge_gap = round(7 * scale)
    wide_card_height = round(59 * scale)
    side_span = round(48 * scale) + round(17 * scale) * (NUM_CARDS // NUM_PLAYERS - 1)
    side_top = round(sy(270) - side_span / 2 + 20 * scale)

    for player in range(NUM_PLAYERS):
        seat = _seat(player, viewer)
        px = panel_columns[seat]
        identity = " · YOU" if player == viewer else ""
        turn = " · TURN" if player == active and not ended else ""
        name = f"P{player}{identity}{turn}"
        metric = (
            f"SCORE {int(scores[player])}  ·  REWARD {float(rewards[player]):+.1f}"
            if ended
            else f"PENALTY {int(penalties[player])}"
        )
        name_box = draw.textbbox((0, 0), name, font=small_font)
        metric_box = draw.textbbox((0, 0), metric, font=small_font)
        pad = round(7 * scale)
        lw = max(name_box[2] - name_box[0], metric_box[2] - metric_box[0]) + 2 * pad
        line_height = max(name_box[3] - name_box[1], metric_box[3] - metric_box[1])
        lh = 2 * line_height + round(5 * scale) + 2 * pad
        fill = "#594915" if turn else "#09231bdc"
        winner = ended and bool(winners[player])
        outline = "#86efac" if winner else "#fbbf24" if turn else "#ffffff35"
        if seat == "top":
            # The top hand sits against the rim, so its badge hangs below it.
            centre_y = sy(91) + wide_card_height + badge_gap + lh / 2
        elif seat == "bottom":
            centre_y = sy(455) - badge_gap - lh / 2
        else:
            centre_y = side_top - badge_gap - lh / 2
        draw.rounded_rectangle(
            (
                round(sx(px) - lw / 2),
                round(centre_y - lh / 2),
                round(sx(px) + lw / 2),
                round(centre_y + lh / 2),
            ),
            radius=round(8 * scale),
            fill=fill,
            outline=outline,
            width=max(1, round(2 * scale)),
        )
        line_offset = (line_height + round(3 * scale)) / 2
        _center(
            draw,
            (sx(px), round(centre_y - line_offset)),
            name,
            small_font,
            "#f8fafc",
        )
        _center(
            draw,
            (sx(px), round(centre_y + line_offset)),
            metric,
            small_font,
            "#93c5fd" if ended else "#d1fae5",
        )

        cards = [int(card) for card in np.flatnonzero(hands[player])]
        hidden = viewer is not None and player != viewer
        ox, oy = hand_origins[seat]
        if seat in ("bottom", "top"):
            card_size = (round(42 * scale), round(59 * scale))
            gap = round((30 if seat == "bottom" else 25) * scale)
            total = card_size[0] + gap * max(len(cards) - 1, 0)
            start_x = round(sx(480) - total / 2)
            for index, card in enumerate(cards):
                origin = (start_x + index * gap, sy(oy))
                if hidden:
                    _card_back(draw, origin, card_size)
                    continue
                _card(
                    draw,
                    card,
                    origin,
                    card_size,
                    (card_font, suit_font),
                    legal=player == active and legal is not None and bool(legal[card]),
                )
        else:
            card_size = (round(34 * scale), round(48 * scale))
            gap = round(17 * scale)
            total = card_size[1] + gap * max(len(cards) - 1, 0)
            start_y = round(sy(270) - total / 2 + 20 * scale)
            for index, card in enumerate(cards):
                origin = (sx(ox), start_y + index * gap)
                if hidden:
                    _card_back(draw, origin, card_size)
                    continue
                _card(
                    draw,
                    card,
                    origin,
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
    visible_trick, leader, _ = _visible_trick(state)
    for offset, raw_card in enumerate(visible_trick):
        card = int(raw_card)
        if card < 0:
            continue
        player = (leader + offset) % NUM_PLAYERS
        tx, ty = trick_positions[_seat(player, viewer)]
        _card(
            draw,
            card,
            (sx(tx), sy(ty)),
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
        _center(draw, (sx(480), sy(303)), result, ui_font, "#86efac")

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
    output = Path(path).expanduser().resolve()
    if output.suffix.lower() != ".gif":
        raise ValueError("GIF output path must use the .gif suffix")
    frames = [render_gif_frame(state, viewer, size) for state in states]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
        format="GIF",
    )
    return output
