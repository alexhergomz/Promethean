"""Database row models. Exercises nested defs (class with methods).
def_span on User.__init__ should not bleed into User.full_name.
"""


class User:
    def __init__(self, row):
        self.id = row["id"]
        self.name = row["name"]

    def full_name(self):
        return self.name.upper()

    def to_dict(self):
        return {"id": self.id, "name": self.name}
