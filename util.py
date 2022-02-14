from typing import List

import numpy as np

from table import Table

relationship_matrix = None
relationship_matrix_cache = relationship_matrix

def reset_matrix(num_of_persons):
    global relationship_matrix
    array = np.random.randint(-5, 6, size=(num_of_persons, num_of_persons))
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


def sparse_matrix(k=5):
    global relationship_matrix,relationship_matrix_cache
    relationship_matrix_cache = relationship_matrix.copy()
    indices = np.argpartition(relationship_matrix, -k, axis=1)[:, -k:].tolist()
    relationship_matrix = np.array([relationship_matrix[i][j] if j in indices[i] else 0 for i, _ in enumerate(indices) for j in
              range(relationship_matrix.shape[0])]).reshape(relationship_matrix.shape)


def normal_matrix():
    global relationship_matrix,relationship_matrix_cache
    relationship_matrix = relationship_matrix_cache.copy()