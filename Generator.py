import json
import numpy as np
import os
import random
import sys
sys.stdout.reconfigure(encoding="utf-8")
def generate_margins_2d(num_suppliers, num_consumers, total=None, balance_margins=True):
    if balance_margins:
        if total is None:
            total = random.randint(50, 200)
        cuts = sorted(random.sample(range(1, total), num_suppliers + num_consumers - 1))
        parts = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, len(cuts))] + [total - cuts[-1]]
        supplies = parts[:num_suppliers]
        demands = parts[num_suppliers:]
    else:
        supplies = [random.randint(0, 100) for _ in range(num_suppliers)]
        demands = [random.randint(0, 100) for _ in range(num_consumers)]
    return supplies, demands
def generate_costs_2d(num_suppliers, num_consumers, low=1, high=100):
    return np.random.randint(low, high+1, size=(num_suppliers, num_consumers)).tolist()
def normalize_costs(costs):
    arr = np.array(costs, dtype=float)
    minc, maxc = arr.min(), arr.max()
    if maxc == minc:
        return arr.tolist(), (minc, maxc)
    norm = (arr - minc) / (maxc - minc)
    return norm.tolist(), (minc, maxc)
def generate_instance_2d(num_suppliers, num_consumers,
                         total=None, balance_margins=True,
                         cost_low=1, cost_high=100,
                         normalize_costs_flag=False,
                         output_path=None):
    supplies, demands = generate_margins_2d(num_suppliers, num_consumers, total, balance_margins)
    costs = generate_costs_2d(num_suppliers, num_consumers, cost_low, cost_high)
    norm_params = None
    if normalize_costs_flag:
        costs, norm_params = normalize_costs(costs)
    instance = {
        "margins": [supplies, demands],
        "costs": costs,
        "balanced_margins": balance_margins,
        "normalized_costs": normalize_costs_flag,
        "normalization_params": norm_params
    }
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(instance, f, ensure_ascii=False, indent=2)
    return instance
def generate_margins_nd(dim_sizes, total=None, balance_margins=True):
    N = len(dim_sizes)
    margins = []
    if balance_margins:
        if total is None:
            total = random.randint(50, 200)
        for size in dim_sizes:
            cuts = sorted(random.sample(range(1, total), size-1))
            parts = [cuts[0]] + [cuts[i]-cuts[i-1] for i in range(1, len(cuts))] + [total-cuts[-1]]
            margins.append(parts)
    else:
        for size in dim_sizes:
            margins.append([random.randint(0, 100) for _ in range(size)])
    return margins
def generate_costs_nd(dim_sizes, low=1, high=100):
    return np.random.randint(low, high+1, size=tuple(dim_sizes)).tolist()
def generate_instance_nd(dim_sizes,
                         total=None, balance_margins=True,
                         cost_low=1, cost_high=100,
                         normalize_costs_flag=False,
                         output_path=None):
    margins = generate_margins_nd(dim_sizes, total, balance_margins)
    costs = generate_costs_nd(dim_sizes, cost_low, cost_high)
    norm_params = None
    if normalize_costs_flag:
        flat = np.array(costs, dtype=float)
        if flat.max() != flat.min():
            flat_norm = (flat - flat.min()) / (flat.max() - flat.min())
        else:
            flat_norm = flat
        costs = flat_norm.tolist()
        norm_params = (float(flat.min()), float(flat.max()))
    instance = {
        "margins": margins,
        "costs": costs,
        "balanced_margins": balance_margins,
        "normalized_costs": normalize_costs_flag,
        "normalization_params": norm_params
    }
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(instance, f, ensure_ascii=False, indent=2)
    return instance
if __name__ == "__main__":
    generate_instance_nd([4,4,4,4], total=500, balance_margins=True,
                                     normalize_costs_flag=False,
                                     output_path="./data/4d_bal.json")