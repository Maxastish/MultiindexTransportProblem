import time
import pandas as pd
import numpy as np
from Generator import generate_instance_nd
from Transport_Problem import (
    genetic_algorithm_islands,
    northwest_seed, least_cost_seed, HiGHS_seed, random_seed
)
import sys
sys.stdout.reconfigure(encoding="utf-8")
# Параметры экспериментов
EXPERIMENTS = [
    {'dims': [100, 100],    'num_tasks': 10, 'runs': 10},
    {'dims': [3, 3, 3, 3, 3, 3, 3],    'num_tasks': 10, 'runs': 10},
    {'dims': [10, 10, 10, 10],    'num_tasks': 10, 'runs': 10},
    {'dims': [3, 3],       'num_tasks': 10, 'runs': 10},
    {'dims': [5, 5],       'num_tasks': 10, 'runs': 10},
    {'dims': [7, 7],       'num_tasks': 10, 'runs': 10},
    {'dims': [3, 3, 3],    'num_tasks': 10, 'runs': 10},
    {'dims': [5, 5, 5],    'num_tasks': 10, 'runs': 10},
    {'dims': [7, 7, 7],    'num_tasks': 10, 'runs': 10},
    {'dims': [3, 3, 3, 3],    'num_tasks': 10, 'runs': 10},
    {'dims': [5, 5, 5, 5],    'num_tasks': 10, 'runs': 10},
]
POP_SIZE = 100
GENERATIONS = 100
ISLANDS = 4
SEED_RATIO = 0.25
MUTATION_RATE = 0.02
SEED_METHODS = [
    ('Northwest', northwest_seed),
    ('LeastCost', least_cost_seed),
    ('LP', HiGHS_seed),
    ('Random', random_seed)
]
def run_experiment(exp, balance_margins=False):
    dims = exp['dims']
    num_tasks = exp['num_tasks']
    runs = exp['runs']
    records = []
    for task_id in range(1, num_tasks + 1):
        instance = generate_instance_nd(dims, balance_margins=False, normalize_costs_flag=False)
        constraints = [np.array(m) for m in instance['margins']]
        costs = np.array(instance['costs'])
        for name, method in SEED_METHODS:
            start = time.time()
            plan = method(constraints, costs)
            t = time.time() - start
            cost_val = float(np.sum(plan * costs))
            records.append({
                'dims': tuple(dims),
                'task_id': task_id,
                'method': name,
                'run_id': 0,
                'best_cost': cost_val,
                'time_sec': t
            })
        for run_id in range(1, runs + 1):
            start = time.time()
            _, best_cost, _ = genetic_algorithm_islands(
                constraints, costs,
                pop_size=POP_SIZE,
                generations=GENERATIONS,
                islands=ISLANDS,
                seed_ratio=SEED_RATIO,
                mutation_rate=MUTATION_RATE
            )
            t = time.time() - start
            records.append({
                'dims': tuple(dims),
                'task_id': task_id,
                'method': 'GA',
                'run_id': run_id,
                'best_cost': float(best_cost),
                'time_sec': t
            })
        print(f"Completed dims={dims}, task {task_id}/{num_tasks}")
    return pd.DataFrame(records)

def main(balance_margins=False):
    all_dfs = []
    for exp in EXPERIMENTS:
        print(f"Running experiment for dims={exp['dims']}")
        df = run_experiment(exp, balance_margins)
        all_dfs.append(df)
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_excel('multi_test_results.xlsx', index=False)
    print("All experiments done. Results saved to multi_test_results.xlsx")

if __name__ == '__main__':
    main(balance_margins=False)