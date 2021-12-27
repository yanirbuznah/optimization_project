from typing import Tuple, List

import numpy as np

np.random.seed(0)
array = np.random.randint(-5, 5, size=(1000, 1000))
relationship_matrix = (array + array.T)
np.fill_diagonal(relationship_matrix, 0)
print(relationship_matrix)


class Table:
    def __init__(self, capacity=10):
        self.score = 0
        self.remaining_capacity = capacity
        self.table = []

    def add_person(self, id):
        if self.remaining_capacity <= 0:
            raise "Table is full"
        self.table.append(id)
        self.remaining_capacity -= 1
        self.score += self.get_score_with_person(id)

    def get_score_with_person(self, id):
        if len(self.table) == 0:
            return 0
        return sum([relationship_matrix[p][id] for p in self.table])

    def __str__(self):
        return self.table.__str__()

    def is_full(self):
        return self.remaining_capacity == 0

def heuristic1(tables: List[Table]):
    for id,p in enumerate(relationship_matrix):
        max_score = -100
        max_table = None
        for i,table in enumerate(tables):
            if table.is_full():
                continue
            score = table.get_score_with_person(id)
            if score > max_score:
                max_score = score
                max_table = table
        max_table.add_person(id)
    return tables


def heuristic2(tables: List[Table]):
    for id,p in enumerate(relationship_matrix):
        max_score = -100
        max_table = None
        for i,table in enumerate(tables):
            if table.is_full():
                continue
            score = table.get_score_with_person(id)
            if score > max_score:
                max_score = score
                max_table = table
        if max_score < 0:
            raise "fail to find good table"
        max_table.add_person(id)
    return tables

if __name__ == '__main__':
    tables = [Table() for x in range(100)]
    heuristic1(tables)
    # for t in tables:
    #     print(t)
    print((sum(t.score for t in tables)))
