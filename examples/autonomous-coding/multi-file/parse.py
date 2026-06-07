"""Parse raw 'date,category,amount' lines into Transaction records."""
from model import Transaction


def parse_line(line):
    date, category, amount = line.strip().split(",")
    # amount is a currency value like "12.50"
    return Transaction(date, category, int(amount))


def parse_lines(text):
    return [parse_line(l) for l in text.strip().splitlines() if l.strip()]
