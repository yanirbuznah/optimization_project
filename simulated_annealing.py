"""
Simulated Annealing Class
"""
import copy
import math
import random
import matplotlib.pyplot as plt

class SimulatedAnnealing:
    def __init__(self, initial_solution, solution_evaluator, initial_temp = 1, final_temp = 0.1,
                 temp_reduction='geometric',
                 not_improve_limit=200,
                 iteration_per_temp=2000, alpha=0.1, beta=5):
        self.solution = initial_solution
        self.evaluate = solution_evaluator
        self.initial_temp = initial_temp
        self.curr_temp = initial_temp
        self.final_temp = final_temp
        self.iteration_per_temp = iteration_per_temp
        self.alpha = alpha
        self.beta = beta
        self.not_improve_limit = not_improve_limit

        if temp_reduction == "linear":
            self.decrement_rule = self.linear_temp_reduction
        elif temp_reduction == "geometric":
            self.decrement_rule = self.geometric_temp_reduction
        elif temp_reduction == "slowDecrease":
            self.decrement_rule = self.slow_decrease_temp_reduction
        else:
            self.decrement_rule = temp_reduction

    def linear_temp_reduction(self):
        self.curr_temp -= self.alpha

    def geometric_temp_reduction(self):
        #change from 1/alpha for compatibility with the linear reduction
        self.curr_temp *= self.alpha

    def slow_decrease_temp_reduction(self):
        self.curr_temp = self.curr_temp / (1 + self.alpha * self.curr_temp)

    def is_termination_criteria_met(self, not_improve):
        # can add more termination criteria
        return self.curr_temp <= self.final_temp or self.not_improve_limit <= not_improve

    # def run(self):
    #     while not self.is_termination_criteria_met():
    #         # iterate that number of times
    #         for i in range(self.iterationPerTemp):
    #             # get all of the neighbors
    #             neighbors = self.neighborOperator(self.solution)
    #             # pick a random neighbor
    #             newSolution = random.choice(neighbors)
    #             # get the cost between the two solutions
    #             cost = self.evaluate(self.solution) - self.evaluate(newSolution)
    #             # if the new solution is better, accept it
    #             if cost >= 0:
    #                 self.solution = newSolution
    #             # if the new solution is not better, accept it with a probability of e^(-cost/temp)
    #             else:
    #                 if random.uniform(0, 1) < math.exp(-cost / self.currTemp):
    #                     self.solution = newSolution
    #         # decrement the temperature
    #         self.decrementRule()

    def run(self,plot = True):
        tables = self.solution
        init_score = (sum(t.score for t in tables))
        best_score = init_score
        best_tables = copy.deepcopy(tables)
        not_improved = 0
        scores = []
        while not self.is_termination_criteria_met(not_improved):
            for i in range(self.iteration_per_temp):

                t1, t2 = random.sample(tables, 2)
                p1 = t1.pick_random_person()
                p2 = t2.pick_random_person()
                delta = (t1.score_after_exchange(p1, p2) + t2.score_after_exchange(p2, p1)) - (t1.score + t2.score)
                current_score = (sum(t.score for t in tables)) + delta
                scores.append(current_score / len(tables))
                try:  # to avoid overflow
                    p = math.exp(delta / self.curr_temp)
                except:
                    p = 2
                if p > random.uniform(0, 1):
                    t1.remove_person(p1)
                    t2.remove_person(p2)
                    t2.add_person(p1)
                    t1.add_person(p2)

                if delta > 0:
                    not_improved = 0
                else:
                    not_improved += 1


                if best_score < current_score:
                    best_score = current_score
                    best_tables = copy.deepcopy(tables)

            self.decrement_rule()

        if plot:
            plt.plot(scores, label='h2f')
            plt.title(f'SA for {len(tables)} tables')
            plt.savefig(f'SA plots/SA for {len(tables)} tables')
            plt.show()
        return copy.deepcopy(best_tables)
