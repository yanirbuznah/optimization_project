from typing import List

import numpy as np

from table import Table

relationship_matrix = None


def reset_matrix(num_of_persons):
    global relationship_matrix
    array = np.random.randint(-5, 5, size=(num_of_persons, num_of_persons))
    relationship_matrix = (array + array.T)
    np.fill_diagonal(relationship_matrix, 0)
    print(relationship_matrix)


def available_tables_for_person(tables: List[Table], person):
    available_tables = 0
    for t in tables:
        if t.is_full():
            continue
        available_tables += int(t.get_score_with_person(person) - t.score >= 0)
    return available_tables
