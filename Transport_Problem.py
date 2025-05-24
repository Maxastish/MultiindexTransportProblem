import json
import random
import threading
import time
import numpy as np
from copy import deepcopy
from scipy.optimize import linprog
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
def read_data_from_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    constraints = [np.array(m) for m in data['margins']]
    costs = np.array(data['costs'])
    return constraints, costs
def northwest_seed(constraints, costs):
    rem, plan = [c.copy() for c in constraints], np.zeros(costs.shape, int)
    shape, idx = costs.shape, [0]*costs.ndim
    while all(idx[d] < shape[d] for d in range(costs.ndim)):
        avail = min(rem[d][idx[d]] for d in range(costs.ndim))
        plan[tuple(idx)] = avail
        for d in range(costs.ndim): rem[d][idx[d]] -= avail
        for d in range(costs.ndim):
            if rem[d][idx[d]] == 0:
                idx[d] += 1
                break
    return plan
def least_cost_seed(constraints, costs):
    rem, plan = [c.copy() for c in constraints], np.zeros(costs.shape, int)
    cells = list(np.ndindex(costs.shape)); cells.sort(key=lambda i: costs[i])
    for i in cells:
        avail = min(rem[d][i[d]] for d in range(costs.ndim))
        if avail > 0:
            plan[i] = avail
            for d in range(costs.ndim): rem[d][i[d]] -= avail
    return plan
def HiGHS_seed(constraints, costs):
    sizes, c = list(costs.shape), costs.flatten()
    A_eq, b_eq = [], []
    for dim, sz in enumerate(sizes):
        for pos in range(sz):
            row = np.zeros(costs.size)
            for idx, coord in enumerate(np.ndindex(*sizes)):
                if coord[dim] == pos:
                    row[idx] = 1
            A_eq.append(row); b_eq.append(constraints[dim][pos])
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)]*c.size, method='highs')
    plan = res.x.reshape(sizes) if res.success else np.zeros(sizes)
    return np.round(plan).astype(int)
def random_seed(constraints, costs):
    rem, plan = [c.copy() for c in constraints], np.zeros(costs.shape, int)
    cells = list(np.ndindex(costs.shape)); random.shuffle(cells)
    for i in cells:
        avail = min(rem[d][i[d]] for d in range(costs.ndim))
        if avail > 0:
            x = random.randint(0, avail); plan[i] = x
            for d in range(costs.ndim): rem[d][i[d]] -= x
    return plan
def violation_summary(plan, constraints):
    total_over = 0
    total_under = 0
    dims = plan.ndim
    axes = list(range(dims))
    for axis in axes:
        proj = plan.sum(axis=tuple(a for a in axes if a != axis))
        diff = proj - constraints[axis]
        total_over  += diff.clip(min=0).sum()
        total_under += (-diff).clip(min=0).sum()
    return int(total_over), int(total_under)
def is_valid(plan, constraints):
    for ax in range(plan.ndim):
        proj = plan.sum(axis=tuple(i for i in range(plan.ndim) if i != ax))
        if np.any(proj > constraints[ax]): return False
    return True
def fitness(plan, costs, constraints, base_over=50, base_under=50, gen=0, total_gens=1):
    over = base_over * (1 + gen/total_gens)
    under = base_under * (1 + gen/total_gens)
    val = np.sum(plan * costs)
    for ax in range(plan.ndim):
        proj = plan.sum(axis=tuple(i for i in range(plan.ndim) if i != ax))
        diff = proj - constraints[ax]
        val += over * np.sum(diff.clip(min=0))
        val += under * np.sum((-diff).clip(min=0))
    return val
def local_hill_climb(plan, costs, constraints, steps=100):
    best, bc = plan.copy(), fitness(plan, costs, constraints)
    for _ in range(steps):
        src = tuple(random.randrange(s) for s in plan.shape)
        dst = tuple(random.randrange(s) for s in plan.shape)
        if best[src] > 0:
            cand = best.copy(); cand[src] -= 1; cand[dst] += 1
            if is_valid(cand, constraints):
                fc = fitness(cand, costs, constraints)
                if fc < bc: best, bc = cand, fc
    return best
def simulated_annealing(plan, costs, constraints, T0=1.0, alpha=0.99, steps=100):
    cur, cc = plan.copy(), fitness(plan,costs,constraints)
    best, bc = cur.copy(), cc; T=T0
    for _ in range(steps):
        idx = tuple(random.randrange(s) for s in plan.shape)
        if cur[idx] > 0:
            cand = cur.copy(); cand[idx] -= 1
            jdx = tuple(random.randrange(s) for s in plan.shape); cand[jdx] += 1
            if is_valid(cand,constraints):
                fc = fitness(cand,costs,constraints); dE = fc-cc
                if dE<0 or random.random()<np.exp(-dE/T): cur,cc = cand,fc
                if fc<bc: best,bc = cand.copy(),fc
        T *= alpha
    return best
def variable_neighborhood(plan, constraints, costs, mutation_rate):
    r = random.choice([0,1,2])
    if r == 0: return mutate(plan, mutation_rate)
    if r == 1: return local_hill_climb(plan, costs, constraints)
    return simulated_annealing(plan, costs, constraints)
def mutate(plan, rate):
    for idx in np.ndindex(*plan.shape):
        if random.random() < rate: plan[idx] = max(0, plan[idx] + random.randint(-1,1))
    return plan
def multi_start(constraints, costs, trials=50):
    best, bc = None, float('inf')
    for _ in range(trials):
        p = random_seed(constraints,costs)
        p = local_hill_climb(p,costs,constraints)
        c = fitness(p,costs,constraints)
        if c < bc: best,bc = p.copy(),c
    return best
def tournament_select(pop, costs, constraints, gen, total_gens, k=3):
    picks = random.sample(pop, k)
    return min(picks, key=lambda p: fitness(p,costs,constraints,gen=gen,total_gens=total_gens))
def crossover(a, b):
    mask = np.random.rand(*a.shape) < 0.5
    return np.where(mask, a, b)
def genetic_algorithm_islands(constraints, costs,
                              pop_size=100, generations=100,
                              islands=4, seed_ratio=0.25, 
                              mutation_rate=0.02, retain_frac=0.5, elitism=5,
                              iso_ratio=0.5, comm_ratio=0.05,
                              update_gui=None, stop_event=None):
    iso_gens = int(generations * iso_ratio)
    comm_int = max(1, int(generations * comm_ratio))
    best0 = multi_start(constraints, costs, 30)
    seeds = [northwest_seed, least_cost_seed, HiGHS_seed, random_seed]
    archi = []
    for isl in range(islands):
        pop = [best0.copy()]
        m = seeds[isl % len(seeds)]; n_seed = int((pop_size-1)*seed_ratio)
        for _ in range(n_seed): pop.append(m(constraints,costs))
        for _ in range(pop_size-1-n_seed): pop.append(random_seed(constraints,costs))
        archi.append(pop)
    logs = [[None]*generations for _ in range(islands)]
    global_best, global_cost = None, float('inf')
    for gen in range(1, generations+1):
        if stop_event and stop_event.is_set(): break
        for isl in range(islands):
            pop = archi[isl]
            valid = [p for p in pop if is_valid(p,constraints)]
            need = int(pop_size*retain_frac) - len(valid)
            for _ in range(max(0, need)): valid.append(random_seed(constraints,costs))
            valid.sort(key=lambda p: fitness(p,costs,constraints,gen, generations))
            elites = valid[:elitism]; children = []
            while len(children) < pop_size-elitism:
                p1 = tournament_select(valid,costs,constraints,gen,generations)
                p2 = tournament_select(valid,costs,constraints,gen,generations)
                c = crossover(p1,p2)
                children.append(variable_neighborhood(c,constraints,costs,mutation_rate))
            archi[isl] = elites + children
            bl = min(archi[isl], key=lambda p: fitness(p,costs,constraints,gen,generations))
            bc = fitness(bl,costs,constraints,gen,generations)
            logs[isl][gen-1] = bc
            over, under = violation_summary(bl, constraints)
            if not (over > 0 and under > 0) and bc < global_cost:
                global_cost, global_best = bc, bl.copy()
        if gen>iso_gens and (gen-iso_gens)%comm_int==0:
            migrants=[]
            for pop in archi:
                pop.sort(key=lambda p: fitness(p,costs,constraints,gen,generations))
                migrants.extend(deepcopy(pop[:elitism]))
            for isl in range(islands):
                archi[isl].sort(key=lambda p: fitness(p,costs,constraints,gen,generations), reverse=True)
                for i in range(elitism): archi[isl][i] = random.choice(migrants)
        if update_gui: update_gui(gen, logs, global_best, global_cost)
    return global_best, global_cost, logs
def export_plan_to_excel(plan, filename):
    shape = plan.shape
    indices = list(np.ndindex(shape))
    data = [list(idx) + [plan[idx]] for idx in indices if plan[idx] > 0]
    cols = [f"Dim{i+1}" for i in range(plan.ndim)] + ["Value"]
    df = pd.DataFrame(data, columns=cols)
    df.to_excel(filename, index=False)
def run_gui():
    root = tk.Tk()
    root.title("Genetic Algo Transport problem")
    root.state("zoomed")
    constraints, costs = None, None
    seed_methods = [northwest_seed, least_cost_seed, HiGHS_seed, random_seed]
    stop_event = threading.Event()
    def on_closing():
        if messagebox.askokcancel("Выход", "Вы действительно хотите выйти?"):
            stop_event.set()
            root.destroy()
            os._exit(0)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    constraints, costs = None, None
    seed_methods = [northwest_seed, least_cost_seed, HiGHS_seed, random_seed]
    stop_event = threading.Event()
    def add_copy_menu(text_widget):
        menu = tk.Menu(text_widget, tearoff=0)
        menu.add_command(label="Copy", command=lambda: text_widget.event_generate("<<Copy>>"))
        text_widget.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))
        text_widget.bind("<Control-c>", lambda e: text_widget.event_generate("<<Copy>>"))
    def load_file():
        nonlocal constraints, costs
        fn = filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if not fn:
            return
        constraints, costs = read_data_from_file(fn)
        ttk.Label(main_frame, text=f"Loaded: {fn}") \
            .grid(row=0, column=3, columnspan=4, pady=(0, 5), sticky='w')
        matrix_text = tk.Text(main_frame, height=30, width=100)
        matrix_text.grid(row=4, column=0, columnspan=4, pady=(5, 10), sticky='nsew')
        add_copy_menu(matrix_text)
        matrix_text.insert(tk.END, "Constraints (margins):\n")
        for i, vec in enumerate(constraints, start=1):
            matrix_text.insert(tk.END, f" dim {i}: {vec.tolist()}\n")
        matrix_text.insert(tk.END, "\nCosts matrix:\n")
        def print_ndarray(arr, prefix=""):
            arr = np.array(arr)
            if arr.ndim <= 2:
                df = pd.DataFrame(arr)
                matrix_text.insert(tk.END, f"{prefix}\n{df.to_string(index=False)}\n\n")
            else:
                for idx, subarr in enumerate(arr):
                    print_ndarray(subarr, prefix=f"{prefix}[{idx}]")
        print_ndarray(costs, prefix="Matrix_")
    ttk.Button(main_frame, text='Load JSON', command=load_file) \
        .grid(row=0, column=0, pady=5, sticky='w')
    def start_algorithm():
        nonlocal stop_event
        try:
            psz = int(pop_entry.get())
            gsz = int(gen_entry.get())
            iso_ratio = float(iso_ratio_entry.get())
            mutation_rate = float(mutation_entry.get())
            isz = 4
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters")
            return
        if constraints is None or costs is None:
            messagebox.showerror("Error", "Load data first")
            return
        for widget in main_frame.winfo_children():
            widget.destroy()
        ttk.Label(main_frame, text="Algorithm running...").grid(row=0, column=0, columnspan=4, pady=10)
        gen_label = ttk.Label(main_frame, text="Gen: 0/0")
        gen_label.grid(row=1, column=0, sticky='w')
        time_label = ttk.Label(main_frame, text="Time: 0.00s")
        time_label.grid(row=1, column=1, sticky='w')
        plan_text = tk.Text(main_frame, height=10, width=60)
        plan_text.grid(row=2, column=0, columnspan=2, pady=10, sticky='nsew')
        add_copy_menu(plan_text)
        cost_text = tk.Text(main_frame, height=10, width=55)
        cost_text.grid(row=2, column=2, columnspan=2, pady=10, sticky='nsew')
        add_copy_menu(cost_text)
        stop_event.clear()
        stop_btn = ttk.Button(main_frame, text="Stop", command=lambda: stop_event.set())
        stop_btn.grid(row=3, column=0, pady=10, sticky='w')
        start_time = time.time()
        def update_generation(gen, logs, best_plan, best_cost):
            elapsed = time.time() - start_time
            gen_label.config(text=f"Gen: {gen}/{gsz}, Best: {best_cost:.1f}")
            time_label.config(text=f"Time: {elapsed:.2f}s")
            plan_text.delete('1.0', tk.END)
            plan_text.insert(tk.END, str(best_plan))
            current_costs = []
            min_costs = []
            for hist in logs:
                vals = [v for v in hist[:gen] if v is not None]
                if vals:
                    current_costs.append(vals[-1])
                    min_costs.append(min(vals))
                else:
                    current_costs.append(float('nan'))
                    min_costs.append(float('nan'))
            df = pd.DataFrame({
                'Island (initial method)': [seed_methods[i % len(seed_methods)].__name__ for i in range(len(logs))],
                'Current Cost': current_costs,
                'Min Cost':     min_costs
            })
            cost_text.delete('1.0', tk.END)
            cost_text.insert(tk.END, df.to_string(index=False))
        def genetic_thread():
            best, cost, logs = genetic_algorithm_islands(
                constraints, costs,
                pop_size=psz,
                generations=gsz,
                islands=isz,
                iso_ratio=iso_ratio,
                mutation_rate=mutation_rate,
                update_gui=update_generation,
                stop_event=stop_event
            )
            clean_logs = [[v for v in isl if v is not None] for isl in logs]
            show_results(best, cost, clean_logs)
        threading.Thread(target=genetic_thread, daemon=True).start()
    def show_results(best_plan, best_cost, logs):
        for widget in main_frame.winfo_children(): widget.destroy()
        ttk.Label(main_frame, text=f"Final Best Cost: {best_cost:.1f}")\
            .grid(row=0, column=0, columnspan=4, pady=10)
        sat_info = []
        for ax in range(best_plan.ndim):
            proj = best_plan.sum(axis=tuple(i for i in range(best_plan.ndim) if i != ax))
            for idx, used in enumerate(proj):
                total = constraints[ax][idx]
                if used == total:
                    status = "Полностью использовано"
                    left = 0
                elif used < total:
                    status = "Не полностью использовано"
                    left = total - used
                else:
                    status = "Превышение!"
                    left = used - total
                sat_info.append((ax+1, idx, used, total, status, left))
        sat_txt = tk.Text(main_frame, height=13, width=100)
        sat_txt.grid(row=1, column=0, columnspan=4, pady=(0,10), sticky='nsew')
        add_copy_menu(sat_txt)
        sat_txt.insert(tk.END, "Проверка потребностей/ресурсов:\n")
        sat_txt.insert(tk.END, f"{'Ось':>3} {'Индекс':>6} {'Использовано':>12} "
                               f"{'Всего':>6} {'Статус':>20} {'Перевышение/_/Остаток':>14}\n")
        for ax, idx, used, total, status, left in sat_info:
            sat_txt.insert(tk.END,
                f"{ax:>3} {idx:>6} {used:>12} {total:>6} {status:>20} {left:>14}\n"
            )
        sat_txt.configure(state='disabled')
        init_results = []
        for method in seed_methods:
            plan0 = method(constraints, costs)
            cost0 = np.sum(plan0 * costs)
            init_results.append({
                'Method': method.__name__,
                'Initial Cost': cost0
            })
        df_init = pd.DataFrame(init_results)
        init_txt = tk.Text(main_frame, height=len(init_results)+2, width=50)
        init_txt.grid(row=5, column=0, columnspan=4, pady=(0,10), sticky='nsew')
        add_copy_menu(init_txt)
        init_txt.insert(tk.END, "Comparison with seed methods:\n")
        init_txt.insert(tk.END, df_init.to_string(index=False))
        init_txt.configure(state='disabled')
        txt = tk.Text(main_frame, height=13)
        txt.grid(row=6, column=0, columnspan=2, sticky='nsew')
        add_copy_menu(txt)
        txt.insert(tk.END, str(best_plan))
        cost_txt = tk.Text(main_frame, height=13, width=60)
        cost_txt.grid(row=6, column=2, columnspan=2, sticky='nsew')
        add_copy_menu(cost_txt)
        final_current = [logs[i][-1] for i in range(len(logs))]
        final_min     = [min(logs[i])    for i in range(len(logs))]
        df_final = pd.DataFrame({
            'Island (init method)': [seed_methods[i % len(seed_methods)].__name__ for i in range(len(logs))],
            'Final Current': final_current,
            'Overall Min':   final_min
        })
        cost_txt.insert(tk.END, df_final.to_string(index=False))
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=2, column=0, columnspan=4, pady=10, sticky='w')
        def export():
            fn = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel','*.xlsx')])
            if not fn: return
            export_plan_to_excel(best_plan, fn)
            messagebox.showinfo('Export', 'Plan exported to ' + fn)
        exp_btn = ttk.Button(btn_frame, text="Export to Excel", command=export)
        exp_btn.grid(row=0, column=0, padx=5)
        options = [seed_methods[i % len(seed_methods)].__name__ for i in range(len(logs))]
        graph_var = tk.StringVar(); graph_var.set(options[0])
        graph_menu = ttk.OptionMenu(btn_frame, graph_var, options[0], *options)
        graph_menu.grid(row=0, column=1, padx=5)
        def show_graph():
            idx = options.index(graph_var.get())
            w = tk.Toplevel(root)
            w.title(f"Cost History - {options[idx]}")
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(logs[idx])
            ax.set_title(f'{options[idx]}')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Cost')
            c = FigureCanvasTkAgg(fig, master=w)
            c.draw(); c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        graph_btn = ttk.Button(btn_frame, text="Show Graph", command=show_graph)
        graph_btn.grid(row=1, column=0, padx=5, pady=5)
        def show_all_graphs():
            w = tk.Toplevel(root); w.title('All Islands Cost Histories')
            fig, axs = plt.subplots(2, 2, figsize=(12,8)); axs = axs.flatten()
            for i, ax in enumerate(axs):
                if i < len(logs):
                    ax.plot(logs[i]); ax.set_title(options[i])
                    ax.set_xlabel('Gen'); ax.set_ylabel('Cost')
                else:
                    ax.axis('off')
            fig.tight_layout()
            c = FigureCanvasTkAgg(fig, master=w); c.draw(); c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        all_btn = ttk.Button(btn_frame, text="Show All Graphs", command=show_all_graphs)
        all_btn.grid(row=1, column=1, padx=5, pady=5)
    ttk.Button(main_frame, text='Load JSON', command=load_file).grid(row=0, column=0, pady=5, sticky='w')
    ttk.Label(main_frame, text='Population size:').grid(row=1, column=0, sticky='e')
    pop_entry = ttk.Entry(main_frame); pop_entry.insert(0,'100'); pop_entry.grid(row=1, column=1, sticky='w')
    ttk.Label(main_frame, text='Generations:').grid(row=1, column=2, sticky='e')
    gen_entry = ttk.Entry(main_frame); gen_entry.insert(0,'100'); gen_entry.grid(row=1, column=3, sticky='w')
    ttk.Label(main_frame, text='Islands rate:').grid(row=2, column=0, sticky='e')
    iso_ratio_entry = ttk.Entry(main_frame); iso_ratio_entry.insert(0,'0.25'); iso_ratio_entry.grid(row=2, column=1, sticky='w')
    ttk.Label(main_frame, text='Mutation rate:').grid(row=2, column=2, sticky='e')
    mutation_entry = ttk.Entry(main_frame); mutation_entry.insert(0,'0.02'); mutation_entry.grid(row=2, column=3, sticky='w')
    start_button = ttk.Button(main_frame, text="Run", command=start_algorithm)
    start_button.grid(row=0, column=2)
    root.mainloop()
if __name__ == '__main__':
    run_gui()
