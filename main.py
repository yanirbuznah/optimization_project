
import time
from typing import Tuple, List
import numpy as np

#np.random.seed(2)
NUM_OF_TABLES = 50
NUM_OF_PERSONS = 500
array = np.random.randint(-5, 5, size=(NUM_OF_PERSONS, NUM_OF_PERSONS))
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


def heuristic1(tables: List[Table]):
    for id, p in enumerate(relationship_matrix):
        max_score = -100
        max_table = None
        for i, table in enumerate(tables):
            if table.is_full():
                continue
            score = table.get_score_with_person(id)
            if score > max_score:
                max_score = score
                max_table = table
        max_table.add_person(id)
    return tables


def heuristic2(tables: List[Table], persons: List[Person], new_tables = False):
    while len(persons) != 0:
        min_tables = len(tables)
        chosen_person = None
        for p in persons:
            p_avaialble = p.available_tables(tables)
            if p_avaialble < min_tables or chosen_person is None:
                min_tables = p_avaialble
                chosen_person = p
        max_score = -10000
        max_table = None
        for i, table in enumerate(tables):
            if table.is_full():
                continue
            score = table.get_score_with_person(chosen_person.id)
            if score > max_score:
                max_score = score
                max_table = table
        if max_score < 0:
            if new_tables:
                tables.append(Table())
                max_table = tables[-1]
                print(f"Failed to find a good table for :{chosen_person.id} open new table, number: {len(tables)}")
            else:
                print(f"Failed to find a good table for :{chosen_person.id}, seat in table:{max_table} with score: {max_score}")
        max_table.add_person(chosen_person.id)
        persons.remove(chosen_person)
    return tables


if __name__ == '__main__':
    h1,h2,h2f = [],[],[]
    for _ in range(100):
        NUM_OF_TABLES = 10
        NUM_OF_PERSONS = 100
        array = np.random.randint(-5, 5, size=(NUM_OF_PERSONS, NUM_OF_PERSONS))
        relationship_matrix = (array + array.T)
        np.fill_diagonal(relationship_matrix, 0)
        tables = [Table() for x in range(NUM_OF_TABLES)]
        persons = [Person(x) for x in range(NUM_OF_PERSONS)]
        start = time.time()
        heuristic1(tables)
        score = (sum(t.score for t in tables))
        h1.append(score)
        print(f"heuristic1:{score}")
        print(time.time() -start)
        tables = [Table() for x in range(NUM_OF_TABLES)]
        start= time.time()
        heuristic2(tables,persons)
        score = (sum(t.score for t in tables))
        h2.append(score)
        print(f"heuristic2:{score}")
        print(time.time() -start)
        tables = [Table() for x in range(NUM_OF_TABLES)]
        persons = [Person(x) for x in range(NUM_OF_PERSONS)]
        start= time.time()
        heuristic2(tables,persons,new_tables=True)

        score = (sum(t.score for t in tables))
        h2f.append(score)
        print(f"heuristic2 with new tables:{score}")
        print(f"Number of tables: {len(tables)}")
        print(time.time() -start)
    print(f"h1 avg:{np.average(h1)}")
    print(f"h2 avg:{np.average(h2)}")
    print(f"h2f avg:{np.average(h2f)}")