import copy
import random
import util

class Table:
    def __init__(self, capacity=10,score= 0,table = None):
        self.score = score
        self.remaining_capacity = capacity
        if not table:
            self.table = []
        else:
            self.table = copy.deepcopy(table)

    def add_person(self, id):
        if self.remaining_capacity <= 0:
            raise "Table is full"
        self.table.append(id)
        self.remaining_capacity -= 1
        self.score += self.get_score_with_person(id)

    def get_score_with_person(self, id):
        if len(self.table) == 0:
            return 0
        return sum(util.relationship_matrix[id][self.table])

    def get_score_without_specific_person(self, id):
        if len(self.table) == 0:
            return 0
        return self.score - sum(util.relationship_matrix[id][self.table])

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
        return self.score - sum(util.relationship_matrix[current_id][self.table]) + sum(
            util.relationship_matrix[other_id][self.table]) - util.relationship_matrix[current_id][other_id]
