import pytest

from heart.cards import (
    CARD_NAMES,
    CLUBS,
    NUM_CARDS,
    QUEEN,
    QUEEN_OF_SPADES,
    SPADES,
    TWO_OF_CLUBS,
    card_id,
    card_name,
)


def test_card_indexing_is_stable():
    assert NUM_CARDS == 52
    assert len(set(CARD_NAMES)) == 52
    assert TWO_OF_CLUBS == 0
    assert QUEEN_OF_SPADES == card_id(SPADES, QUEEN)
    assert card_name(TWO_OF_CLUBS) == "2♣"
    assert card_name(QUEEN_OF_SPADES) == "Q♠"


def test_card_index_validation():
    with pytest.raises(ValueError):
        card_id(CLUBS, 13)
    with pytest.raises(ValueError):
        card_name(52)


@pytest.mark.parametrize("value", [True, 0.0, "0"])
def test_card_index_rejects_non_integer_values(value):
    with pytest.raises(TypeError):
        card_id(value, 0)
    with pytest.raises(TypeError):
        card_id(0, value)
    with pytest.raises(TypeError):
        card_name(value)
