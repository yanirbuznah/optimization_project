import copy
import math
import random
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import util
# np.random.seed(2)
from simulated_annealing import SimulatedAnnealing
from table import Table


def heuristic1(tables: List[Table]):
    for p_id, p in enumerate(util.relationship_matrix):
        max_score = -float('inf')
        max_table = None
        for i, table in enumerate(tables):
            if table.is_full():
                continue
            score = table.get_score_with_person(p_id)
            if score > max_score:
                max_score = score
                max_table = table
        max_table.add_person(p_id)
    return tables


def heuristic2(tables: List[Table], persons, new_tables=False):
    w_color = '\033[91m'
    end_color = '\033[0m'
    while len(persons) != 0:
        min_tables = len(tables)
        chosen_person = None
        for p in persons:
            p_available = util.available_tables_for_person(tables, p)
            if p_available < min_tables or chosen_person is None:
                min_tables = p_available
                chosen_person = p
        max_score = -float('inf')
        max_table = None
        for i, table in enumerate(tables):
            if table.is_full():
                continue
            score = table.get_score_with_person(chosen_person)
            if score > max_score:
                max_score = score
                max_table = table
        if max_score < 0:
            if new_tables:
                tables.append(Table())
                max_table = tables[-1]
                print(
                    f"{w_color}Failed to find a good table for :{chosen_person} open new table, number: {len(tables)}{end_color}")
            else:
                print(
                    f"{w_color}Failed to find a good table for :{chosen_person}, seat in table:{max_table} with score: {max_score}{end_color}")
        max_table.add_person(chosen_person)
        persons.remove(chosen_person)
    return tables


def heuristic3(tables: List[Table]):
    for p_id, p in enumerate(util.relationship_matrix):
        max_score = -float('inf')
        max_table = None
        for i, table in enumerate(tables):
            if table.is_full():
                continue
            score = table.get_score_with_person(p_id) - table.score
            if score > max_score:
                max_score = score
                max_table = table
        max_table.add_person(p_id)
    return tables


def heuristic4(tables: List[Table], persons, new_tables=False):
    w_color = '\033[91m'
    end_color = '\033[0m'
    while len(persons) != 0:
        min_tables = len(tables)
        chosen_person = None
        for p in persons:
            p_available = util.available_tables_for_person(tables, p)
            if p_available < min_tables or chosen_person is None:
                min_tables = p_available
                chosen_person = p
        max_score = -float('inf')
        max_table = None
        for i, table in enumerate(tables):
            if table.is_full():
                continue
            score = table.get_score_with_person(chosen_person) - table.score
            if score > max_score:
                max_score = score
                max_table = table
        if max_score < 0:
            if new_tables:
                tables.append(Table())
                max_table = tables[-1]
                print(
                    f"{w_color}Failed to find a good table for :{chosen_person} open new table, number: {len(tables)}{end_color}")
            else:
                print(
                    f"{w_color}Failed to find a good table for :{chosen_person}, seat in table:{max_table} with score: {max_score}{end_color}")
        max_table.add_person(chosen_person)
        persons.remove(chosen_person)
    return tables


def random_restart(available_tables: List[Table], persons):
    # unseated_persons = persons[:]

    fully_tables = []
    for p in persons:
        table = random.choice(available_tables)
        table.add_person(p)
        if table.is_full():
            t = available_tables.pop(available_tables.index(table))
            fully_tables.append(t)
    return fully_tables


def simulated_annealing(tables: List[Table], persons, not_improve_limit=15, random_start=True):
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


"""
compare between h3 h4 and SA and h3,h4 followed by SA 
"""

def combine_approaches():
    sa_avg, h3_avg, h4_avg, h3_sa_avg, h4_sa_avg = [], [], [], [], []
    sa_avgt, h3_avgt, h4_avgt, h3_sa_avgt, h4_sa_avgt = [], [], [], [], []
    n_tables = [5 * (i + 1) for i in range(20)]
    for n_t in n_tables:
        num_of_tables = n_t
        num_of_persons = num_of_tables * 10
        sa, h3, h4, h3_sa, h4_sa = [], [], [], [], [],
        sat, h3t, h4t, h3_sat, h4_sat = [], [], [], [], []
        for i in range(10):
            util.reset_matrix(num_of_persons)

            tables = [Table() for _ in range(num_of_tables)]
            persons = [x for x in range(num_of_persons)]
            start = time.time()
            tables = heuristic3(tables)
            score = (sum(t.score for t in tables) / n_t)
            h3.append(score)
            print(f"heuristic3:{score}")
            t = time.time() - start
            h3t.append(t)
            print(t)

            start = time.time()
            sim_ann = SimulatedAnnealing(tables, None, initial_temp=1, alpha=0.01, iteration_per_temp=4000,
                                         not_improve_limit=400, temp_reduction='slowDecrease')
            tables = sim_ann.run(plot=False)
            score = (sum(t.score for t in tables) / n_t)
            h3_sa.append(score)
            print(f"simulated annealing after h3:{score}")
            t = time.time() - start
            h3_sat.append(t)
            print(t)

            tables = [Table() for _ in range(num_of_tables)]
            persons = [x for x in range(num_of_persons)]
            start = time.time()
            tables = heuristic4(tables, persons)
            score = (sum(t.score for t in tables) / n_t)
            h4.append(score)
            print(f"heuristic4:{score}")
            t = time.time() - start
            h4t.append(t)
            print(t)

            start = time.time()
            sim_ann = SimulatedAnnealing(tables, None, initial_temp=1, alpha=0.01, iteration_per_temp=4000,
                                         not_improve_limit=400, temp_reduction='slowDecrease')
            tables = sim_ann.run(plot=False)
            score = (sum(t.score for t in tables) / n_t)
            h4_sa.append(score)
            print(f"simulated annealing after h4:{score}")
            t = time.time() - start
            h4_sat.append(t)
            print(t)

            tables = [Table() for _ in range(num_of_tables)]
            persons = [x for x in range(num_of_persons)]
            start = time.time()
            tables = random_restart(tables, persons)
            sim_ann = SimulatedAnnealing(tables, None, initial_temp=1, alpha=0.01, iteration_per_temp=4000,
                                         not_improve_limit=400, temp_reduction='slowDecrease')
            if i % 10 == 0:
                tables = sim_ann.run(plot=True)
            else:
                tables = sim_ann.run(plot=False)
            score = (sum(t.score for t in tables) / n_t)
            sa.append(score)
            print(f"simulated annealing:{score}")
            t = time.time() - start
            sat.append(t)
            print(t)

        sa_avg.append(np.average(sa))
        h3_avg.append(np.average(h3))
        h3_sa_avg.append(np.average(h3_sa))
        h4_avg.append(np.average(h4))
        h4_sa_avg.append(np.average(h4_sa))

        sa_avgt.append(np.average(sat))
        h3_avgt.append(np.average(h3t))
        h3_sa_avgt.append(np.average(h3_sat))
        h4_avgt.append(np.average(h4t))
        h4_sa_avgt.append(np.average(h4_sat))

        print(f"h3 avg:{h3_avg[-1]}")
        print(f"h4 avg:{h4_avg[-1]}")
        print(f"simulated annealing avg:{sa_avg[-1]}")
        print(f"simulated annealing avg after h3:{h3_sa_avg[-1]}")
        print(f"simulated annealing avg after h4:{h4_sa_avg[-1]}")

        print(f"h3 avg time:{h3_avgt[-1]}")
        print(f"h4 avg time:{h4_avgt[-1]}")
        print(f"simulated annealing avg time:{sa_avgt[-1]}")
        print(f"simulated annealing avg time after h3:{h3_sa_avgt[-1]}")
        print(f"simulated annealing avg time after h4:{h4_sa_avgt[-1]}")

    # plt.plot(n_tables, bests_avg, label='best')

    plt.plot(n_tables, h3_avg, '-', label='h3')
    plt.plot(n_tables, h4_avg, '-', label='h4')
    plt.plot(n_tables, sa_avg, '-', label='SA')
    plt.plot(n_tables, h3_sa_avg, '-', label='SA & h3')
    plt.plot(n_tables, h4_sa_avg, '-', label='SA & h4')
    plt.title("Scores")
    plt.legend()
    plt.savefig('scores.png')
    plt.show()

    plt.plot(n_tables, h3_avgt, '-', label='h3')
    plt.plot(n_tables, h4_avgt, '-', label='h4')
    plt.plot(n_tables, sa_avgt, '-', label='SA')
    plt.plot(n_tables, h3_sa_avgt, '-', label='SA & h3')
    plt.plot(n_tables, h4_sa_avgt, '-', label='SA & h4')
    plt.title('Time')
    plt.legend()
    plt.savefig('times.png')
    plt.show()


"""
compare between h3 h4 and SA with 4000 iterations per temperature and 400 iterations without improvement. 
"""


def compare_top3_approaches():
    sa_avg, h3_avg, h4_avg = [], [], []
    sa_avgt, h3_avgt, h4_avgt = [], [], []
    tables_2f_avg, tables_4f_avg = [], []
    n_tables = [5 * (i + 1) for i in range(20)]
    for n_t in n_tables:
        num_of_tables = n_t
        num_of_persons = num_of_tables * 10
        sa, h3, h4 = [], [], [],
        sat, h3t, h4t = [], [], []
        for i in range(10):
            util.reset_matrix(num_of_persons)

            tables = [Table() for _ in range(num_of_tables)]
            persons = [x for x in range(num_of_persons)]
            start = time.time()
            tables = heuristic3(tables)
            score = (sum(t.score for t in tables) / n_t)
            h3.append(score)
            print(f"heuristic3:{score}")
            t = time.time() - start
            h3t.append(t)
            print(t)

            tables = [Table() for _ in range(num_of_tables)]
            persons = [x for x in range(num_of_persons)]
            start = time.time()
            tables = heuristic4(tables, persons)
            score = (sum(t.score for t in tables) / n_t)
            h4.append(score)
            print(f"heuristic4:{score}")
            t = time.time() - start
            h4t.append(t)
            print(t)

            tables = [Table() for _ in range(num_of_tables)]
            persons = [x for x in range(num_of_persons)]
            start = time.time()
            tables = random_restart(tables, persons)
            sim_ann = SimulatedAnnealing(tables, None, initial_temp=1, alpha=0.01, iteration_per_temp=4000,
                                         not_improve_limit=400, temp_reduction='slowDecrease')
            if i % 10 == 0:
                tables = sim_ann.run(plot=True)
            else:
                tables = sim_ann.run(plot=False)
            score = (sum(t.score for t in tables) / n_t)
            sa.append(score)
            print(f"simulated annealing:{score}")
            t = time.time() - start
            sat.append(t)
            print(t)

        sa_avg.append(np.average(sa))
        h3_avg.append(np.average(h3))
        h4_avg.append(np.average(h4))

        sa_avgt.append(np.average(sat))
        h3_avgt.append(np.average(h3t))
        h4_avgt.append(np.average(h4t))

        print(f"h3 avg:{h3_avg[-1]}")
        print(f"h4 avg:{h4_avg[-1]}")
        print(f"simulated annealing avg:{sa_avg[-1]}")

        print(f"h3 avg time:{h3_avgt[-1]}")
        print(f"h4 avg time:{h4_avgt[-1]}")
        print(f"simulated annealing avg time:{sa_avgt[-1]}")

    # plt.plot(n_tables, bests_avg, label='best')

    plt.plot(n_tables, h3_avg, '-o', label='h3')
    plt.plot(n_tables, h4_avg, '-o', label='h4')
    plt.plot(n_tables, sa_avg, '-o', label='simulated annealing')
    plt.title("Scores")
    plt.legend()
    plt.savefig('scores.png')
    plt.show()

    plt.plot(n_tables, h3_avgt, '-o', label='h3')
    plt.plot(n_tables, h4_avgt, '-o', label='h4')
    plt.plot(n_tables, sa_avgt, '-o', label='simulated annealing')
    plt.title('Time')
    plt.legend()
    plt.savefig('times.png')
    plt.show()


"""
plots generator for the report 
"""


def compare_approaches():
    sa_avg, h1_avg, h2_avg, h2f_avg, bests_avg, h3_avg, h4_avg, h4f_avg = [], [], [], [], [], [], [], []
    sa_avgt, h1_avgt, h2_avgt, h2f_avgt, h3_avgt, h4_avgt, h4f_avgt = [], [], [], [], [], [], []
    tables_2f_avg, tables_4f_avg = [], []
    n_tables = [5 * (i + 1) for i in range(20)]
    for n_t in n_tables:
        num_of_tables = n_t
        num_of_persons = num_of_tables * 10
        sa, h1, h2, h2f, bests, h3, h4, h4f = [], [], [], [], [], [], [], []
        sat, h1t, h2t, h2ft, h3t, h4t, h4ft = [], [], [], [], [], [], []
        tables_2f, tables_4f = [], []
        for i in range(10):
            util.reset_matrix(num_of_persons)
            bests.append(np.sum(np.partition(util.relationship_matrix, -10, axis=0)[-10:]) / (2 * n_t))
            print(f"best:{bests[-1]}")

            tables = [Table() for _ in range(num_of_tables)]
            persons = [x for x in range(num_of_persons)]
            start = time.time()
            tables = heuristic1(tables)
            score = (sum(t.score for t in tables) / n_t)
            h1.append(score)
            print(f"heuristic1:{score}")
            t = time.time() - start
            h1t.append(t)
            print(t)

            tables = [Table() for _ in range(num_of_tables)]
            start = time.time()
            tables = heuristic2(tables, persons)
            score = (sum(t.score for t in tables) / n_t)
            h2.append(score)
            print(f"heuristic2:{score}")
            t = time.time() - start
            h2t.append(t)
            print(t)

            tables = [Table() for _ in range(num_of_tables)]
            persons = [x for x in range(num_of_persons)]
            start = time.time()
            tables = heuristic2(tables, persons, new_tables=True)
            score = (sum(t.score for t in tables) / n_t)
            h2f.append(score)
            print(f"heuristic2 with new tables:{score}")
            print(f"Number of tables: {len(tables)}")
            tables_2f.append(len(tables))
            t = time.time() - start
            h2ft.append(t)
            print(t)

            tables = [Table() for _ in range(num_of_tables)]
            persons = [x for x in range(num_of_persons)]
            start = time.time()
            tables = heuristic3(tables)
            score = (sum(t.score for t in tables) / n_t)
            h3.append(score)
            print(f"heuristic3:{score}")
            t = time.time() - start
            h3t.append(t)
            print(t)

            tables = [Table() for _ in range(num_of_tables)]
            persons = [x for x in range(num_of_persons)]
            start = time.time()
            tables = heuristic4(tables, persons)
            score = (sum(t.score for t in tables) / n_t)
            h4.append(score)
            print(f"heuristic4:{score}")
            t = time.time() - start
            h4t.append(t)
            print(t)

            tables = [Table() for _ in range(num_of_tables)]
            persons = [x for x in range(num_of_persons)]
            start = time.time()
            tables = heuristic4(tables, persons, new_tables=True)
            score = (sum(t.score for t in tables) / n_t)
            h4f.append(score)
            print(f"heuristic2 with new tables:{score}")
            print(f"Number of tables: {len(tables)}")
            tables_4f.append(len(tables))
            t = time.time() - start
            h4ft.append(t)
            print(t)

            tables = [Table() for _ in range(num_of_tables)]
            persons = [x for x in range(num_of_persons)]
            start = time.time()
            tables = random_restart(tables, persons)
            sim_ann = SimulatedAnnealing(tables, None, initial_temp=1, alpha=0.01, temp_reduction='slowDecrease')
            if i % 10 == 0:
                tables = sim_ann.run(plot=True)
            else:
                tables = sim_ann.run(plot=False)
            score = (sum(t.score for t in tables) / n_t)
            sa.append(score)
            print(f"simulated annealing:{score}")
            t = time.time() - start
            sat.append(t)
            print(t)

        sa_avg.append(np.average(sa))
        h1_avg.append(np.average(h1))
        h2_avg.append(np.average(h2))
        h2f_avg.append(np.average(h2f))
        h3_avg.append(np.average(h3))
        h4_avg.append(np.average(h4))
        h4f_avg.append(np.average(h4f))
        bests_avg.append(np.average(bests))

        sa_avgt.append(np.average(sat))
        h1_avgt.append(np.average(h1t))
        h2_avgt.append(np.average(h2t))
        h2f_avgt.append(np.average(h2ft))
        h3_avgt.append(np.average(h3t))
        h4_avgt.append(np.average(h4t))
        h4f_avgt.append(np.average(h4ft))

        tables_2f_avg.append(np.average(tables_2f))
        tables_4f_avg.append(np.average(tables_4f))

        print(f"h1 avg:{h1_avg[-1]}")
        print(f"h2 avg:{h2_avg[-1]}")
        print(f"h2f avg:{h2f_avg[-1]}")
        print(f"h3 avg:{h3_avg[-1]}")
        print(f"h4 avg:{h4_avg[-1]}")
        print(f"h4f avg:{h4f_avg[-1]}")
        print(f"simulated annealing avg:{sa_avg[-1]}")
        print(f"best avg:{bests_avg[-1]}")

        print(f"h1 avg time:{h1_avgt[-1]}")
        print(f"h2 avg time:{h2_avgt[-1]}")
        print(f"h2f avg time:{h2f_avgt[-1]}")
        print(f"h3 avg time:{h3_avgt[-1]}")
        print(f"h4 avg time:{h4_avgt[-1]}")
        print(f"h4f avg time:{h4f_avgt[-1]}")
        print(f"simulated annealing avg time:{sa_avgt[-1]}")

    # plt.plot(n_tables, bests_avg, label='best')
    plt.plot(n_tables, h1_avg, '-o', label='h1')
    plt.plot(n_tables, h2_avg, '-o', label='h2')
    plt.plot(n_tables, h2f_avg, '-o', label='h2f')
    plt.plot(n_tables, h3_avg, '-o', label='h3')
    plt.plot(n_tables, h4_avg, '-o', label='h4')
    plt.plot(n_tables, h4f_avg, '-o', label='h4f')
    plt.plot(n_tables, sa_avg, '-o', label='simulated annealing')
    plt.title("Scores")
    plt.legend()
    plt.savefig('scores.png')
    plt.show()

    plt.plot(n_tables, h1_avgt, '-o', label='h1')
    plt.plot(n_tables, h2_avgt, '-o', label='h2')
    plt.plot(n_tables, h2f_avgt, '-o', label='h2f')
    plt.plot(n_tables, h3_avgt, '-o', label='h3')
    plt.plot(n_tables, h4_avgt, '-o', label='h4')
    plt.plot(n_tables, h4f_avgt, '-o', label='h4f')
    plt.plot(n_tables, sa_avgt, '-o', label='simulated annealing')
    plt.title('Time')
    plt.legend()
    plt.savefig('times.png')
    plt.show()

    plt.plot(n_tables, tables_2f_avg, '-o', label='h2f')
    plt.plot(n_tables, tables_4f_avg, '-o', label='h4f')
    plt.title('Num of tables')
    plt.legend()
    plt.savefig('num of tables.png')
    plt.show()


"""
for excel sheets of the SA algorithm
simulated annealing tuning parameters and write to excel files the results
"""


def simulated_annealing_tuning_parameters():
    n_tables = [5 * (i + 1) for i in range(20)]
    temps = [0.1 * (i + 1) for i in range(10)]
    alphas = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    df_list = {'l': [], 's': [], 'g': []}

    for num_of_tables in n_tables:
        d_linear = pd.DataFrame(0, index=temps, columns=alphas)
        d_linear.index.name = 'temperature'
        d_linear.columns.name = 'alpha'
        d_geometric = pd.DataFrame(0, index=temps, columns=alphas)
        d_geometric.index.name = 'temperature'
        d_geometric.columns.name = 'alpha'
        d_slow_decrease = pd.DataFrame(0, index=temps, columns=alphas)
        d_slow_decrease.index.name = 'temperature'
        d_slow_decrease.columns.name = 'alpha'

        num_of_persons = num_of_tables * 10
        for t in temps:
            for a in alphas:
                l_scores, s_scores, g_scores = [], [], []
                for _ in range(10):
                    util.reset_matrix(num_of_persons)
                    tables = [Table() for _ in range(num_of_tables)]
                    persons = [x for x in range(num_of_persons)]
                    tables = random_restart(tables, persons)
                    sim_ann = SimulatedAnnealing(tables, None, t, alpha=a, temp_reduction='linear')
                    tables = sim_ann.run()
                    score = (sum(t.score for t in tables) / num_of_tables)
                    print(f"n_t:{num_of_tables} t:{t}, alpha:{a} {score}")
                    l_scores.append(score)
                    tables = [Table() for _ in range(num_of_tables)]
                    persons = [x for x in range(num_of_persons)]
                    tables = random_restart(tables, persons)
                    sim_ann = SimulatedAnnealing(tables, None, t, alpha=a, temp_reduction='geometric')
                    tables = sim_ann.run()
                    score = (sum(t.score for t in tables) / num_of_tables)
                    print(f"n_t:{num_of_tables} t:{t}, alpha:{a} {score}")
                    g_scores.append(score)
                    tables = [Table() for _ in range(num_of_tables)]
                    persons = [x for x in range(num_of_persons)]
                    tables = random_restart(tables, persons)
                    sim_ann = SimulatedAnnealing(tables, None, t, alpha=a, temp_reduction='slowDecrease')
                    tables = sim_ann.run()
                    score = (sum(t.score for t in tables) / num_of_tables)
                    print(f"n_t:{num_of_tables} t:{t}, alpha:{a} {score}")
                    s_scores.append(score)
                l_avg = np.average(l_scores)
                g_avg = np.average(g_scores)
                s_avg = np.average(s_scores)
                d_linear.loc[t, a] = round(l_avg, 2)
                d_geometric.loc[t, a] = round(g_avg, 2)
                d_slow_decrease.loc[t, a] = round(s_avg, 2)
        df_list['l'].append(d_linear.copy(True))
        df_list['s'].append(d_slow_decrease.copy(True))
        df_list['g'].append(d_geometric.copy(True))

    writer = pd.ExcelWriter('linear.xlsx')
    linear = df_list['l']
    for i, df in enumerate(linear):
        df.to_excel(writer, f'sheet{n_tables[i]}')
    writer.save()
    writer = pd.ExcelWriter('geometric.xlsx')
    geometric = df_list['g']
    for i, df in enumerate(geometric):
        df.to_excel(writer, f'sheet{n_tables[i]}')
    writer.save()
    writer = pd.ExcelWriter('first results/slowDecrease.xlsx')
    slow_decrease = df_list['s']
    for i, df in enumerate(slow_decrease):
        df.to_excel(writer, f'sheet{n_tables[i]}')
    writer.save()


if __name__ == '__main__':

    # simulated_annealing_tuning_parameters()
    #
    # combine_approaches()
    #
    # compare_top3_approaches()
    #
    combine_approaches()
