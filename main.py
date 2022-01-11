import math
import random
import time
from typing import List
import copy

import matplotlib.pyplot as plt
import numpy as np

# np.random.seed(2)
NUM_OF_TABLES = 50
NUM_OF_PERSONS = 500
array = np.random.randint(-5, 5, size=(NUM_OF_PERSONS, NUM_OF_PERSONS))
relationship_matrix = (array + array.T)
np.fill_diagonal(relationship_matrix, 0)
print(relationship_matrix)


class Table:
    def __init__(self, capacity=10,score= 0,table = None):
        self.score = score
        self.remaining_capacity = capacity
        if not table:
            self.table = []
        else:
            self.table = table[:]

    def add_person(self, id):
        if self.remaining_capacity <= 0:
            raise "Table is full"
        self.table.append(id)
        self.remaining_capacity -= 1
        self.score += self.get_score_with_person(id)

    def get_score_with_person(self, id):
        if len(self.table) == 0:
            return 0
        return sum(relationship_matrix[id][self.table])

    def get_score_without_specific_person(self, id):
        if len(self.table) == 0:
            return 0
        return self.score - sum(relationship_matrix[id][self.table])

    def __str__(self):
        return self.table.__str__()

    def is_full(self):
        return self.remaining_capacity == 0

    def remove_person(self, id):
        self.table.remove(id)
        self.remaining_capacity += 1
        self.score -= self.get_score_with_person(id)

    def pick_random_person(self):
        return random.choice(self.table)

    def score_after_exchange(self, current_id, other_id):
        return self.score - sum(relationship_matrix[current_id][self.table]) + sum(
            relationship_matrix[other_id][self.table]) - relationship_matrix[current_id][other_id]
    #
    # def __copy__(self):
    #     return Table(capacity=self.remaining_capacity, score=self.score, table=self.table[:])


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


def heuristic2(tables: List[Table], persons: List[Person], new_tables=False):
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
                print(
                    f"Failed to find a good table for :{chosen_person.id}, seat in table:{max_table} with score: {max_score}")
        max_table.add_person(chosen_person.id)
        persons.remove(chosen_person)
    return tables


def random_restart(available_tables: List[Table], persons: List[Person]):
    # unseated_persons = persons[:]

    fully_tables = []
    for p in persons:
        table = random.choice(available_tables)
        table.add_person(p.id)
        p.seated = True
        if table.is_full():
            t = available_tables.pop(available_tables.index(table))
            fully_tables.append(t)
    return fully_tables


def simulated_annealing(tables: List[Table], persons: List[Person], not_improve_limit = 15,random_start = True):
    if random_start:
        tables = random_restart(tables, persons)
    init_score = (sum(t.score for t in tables))
    best_score = init_score
    best_tables = copy.deepcopy(tables)
    not_improved = 0
    iter = 0
    while not_improved < not_improve_limit or iter <= 10000:
        iter += 1
        t1, t2 = random.sample(tables, 2)
        p1 = t1.pick_random_person()
        p2 = t2.pick_random_person()
        delta = (t1.score_after_exchange(p1, p2) + t2.score_after_exchange(p2, p1)) - (t1.score + t2.score)
        p = math.exp(delta / iter)
        if p > random.uniform(0, 1):
            t1.remove_person(p1)
            t2.remove_person(p2)
            t2.add_person(p1)
            t1.add_person(p2)

        if delta > 0:
            not_improved = 0
        else:
            not_improved += 1
        current_score = (sum(t.score for t in tables))
        if best_score < current_score:
            best_score = current_score
            best_tables = copy.deepcopy(tables)

    final_score = (sum(t.score for t in tables))
    return copy.deepcopy(best_tables)


def main():
    global NUM_OF_TABLES, NUM_OF_PERSONS, array, relationship_matrix
    sa,h1, h2, h2f, bests = [], [], [], [], []
    sa_avg, h1_avg, h2_avg, h2f_avg, bests_avg = [], [], [], [], []
    num_of_tables = [10] #[5 * (i + 1) for i in range(20)]
    for n_t in num_of_tables:
        for _ in range(100):
            NUM_OF_TABLES = n_t
            NUM_OF_PERSONS = 10 * n_t
            array = np.random.randint(-5, 6, size=(NUM_OF_PERSONS, NUM_OF_PERSONS))
            relationship_matrix = (array + array.T)
            np.fill_diagonal(relationship_matrix, 0)
            bests.append(np.sum([np.max(relationship_matrix, axis=0)]) / n_t)
            print(f"best:{bests[-1]}")
            tables = [Table() for x in range(NUM_OF_TABLES)]
            persons = [Person(x) for x in range(NUM_OF_PERSONS)]
            start = time.time()
            tables = heuristic1(tables)
            score = (sum(t.score for t in tables) / n_t)
            h1.append(score)
            print(f"heuristic1:{score}")
            print(time.time() - start)
            tables = [Table() for x in range(NUM_OF_TABLES)]
            start = time.time()
            tables = heuristic2(tables, persons)
            score = (sum(t.score for t in tables) / n_t)
            h2.append(score)
            print(f"heuristic2:{score}")
            print(time.time() - start)
            tables = [Table() for x in range(NUM_OF_TABLES)]
            persons = [Person(x) for x in range(NUM_OF_PERSONS)]
            start = time.time()
            tables = heuristic2(tables, persons, new_tables=True)
            score = (sum(t.score for t in tables) / n_t)
            h2f.append(score)
            print(f"heuristic2 with new tables:{score}")
            print(f"Number of tables: {len(tables)}")
            print(time.time() - start)
            #tables = [Table() for x in range(NUM_OF_TABLES)]
            #persons = [Person(x) for x in range(NUM_OF_PERSONS)]
            start = time.time()
            tables = simulated_annealing(tables, persons,random_start=False,not_improve_limit=20)
            score = (sum(t.score for t in tables) / n_t)
            sa.append(score)
            print(f"simulated annealing:{score}")
            print(time.time() - start)

        sa_avg.append(np.average(sa))
        h1_avg.append(np.average(h1))
        h2_avg.append(np.average(h2))
        h2f_avg.append(np.average(h2f))
        bests_avg.append(np.average(bests))

        print(f"h1 avg:{h1_avg[-1]}")
        print(f"h2 avg:{h2_avg[-1]}")
        print(f"h2f avg:{h2f_avg[-1]}")
        print(f"simulated annealing avg:{sa_avg[-1]}")
        print(f"best avg:{bests_avg[-1]}")
    plt.plot(num_of_tables, bests_avg, label='best')
    plt.plot(num_of_tables, h1_avg, label='h1')
    plt.plot(num_of_tables, h2_avg, label='h2')
    plt.plot(num_of_tables, h2f_avg, label='h2f')
    plt.plot(num_of_tables, sa_avg, label='simulated annealing')
    plt.legend()
    plt.savefig('plot.png')
    plt.show()


if __name__ == '__main__':
    # tables = [Table() for x in range(NUM_OF_TABLES)]
    # persons = [Person(x) for x in range(NUM_OF_PERSONS)]
    # tables = simulated_annealing(tables, persons,not_improve_limit = 15)
    main()
