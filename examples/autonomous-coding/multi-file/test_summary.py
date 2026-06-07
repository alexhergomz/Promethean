from parse import parse_lines
from summary import total, by_category

DATA = """2024-01-01,food,12.50
2024-01-02,food,7.25
2024-01-03,transport,20.00"""


def test_total():
    assert total(parse_lines(DATA)) == 39.75


def test_by_category():
    assert by_category(parse_lines(DATA)) == {"food": 19.75, "transport": 20.0}
