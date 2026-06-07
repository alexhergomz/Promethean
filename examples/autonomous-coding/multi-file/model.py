"""Domain model."""
from collections import namedtuple

Transaction = namedtuple("Transaction", ["date", "category", "amount"])
