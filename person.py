from typing import List

from table import Table


class Person:
    def __init__(self, id):
        self.current_table = None
        self.seated = False
        self.id = id
        # self.evaluations = []

    def is_seated(self):
        return self.seated

    def seat_on_table(self, table):
        self.available_tables -= self.available_tables
        self.seated = True
        self.current_table = table

    def available_tables(self, tables: List[Table]):
        available_tables = 0
        for t in tables:
            if t.is_full():
                continue
            available_tables += int(t.get_score_with_person(self.id) - t.score >= 0)
        return available_tables
