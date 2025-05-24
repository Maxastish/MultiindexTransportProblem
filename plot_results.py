import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
import sys
sys.stdout.reconfigure(encoding="utf-8")

def load_data(path='1.xlsx'):
    """
    Загружает результаты сравнения из Excel-файла.
    """
    df = pd.read_excel(path)
    return df


def summarize(df):
    """
    Вычисляет средние метрики best_cost и time_sec по методам,
    сортирует по avg_cost и добавляет отклонение от минимального значения.
    """
    summary = df.groupby('method').agg(
        avg_cost=('best_cost', 'mean'),
        avg_time=('time_sec', 'mean')
    ).reset_index()
    # Сортировка по стоимости
    summary = summary.sort_values('avg_cost').reset_index(drop=True)
    min_cost = summary['avg_cost'].min()
    summary['deviation'] = summary['avg_cost'] - min_cost
    return summary


def plot_bar(summary):
    """
    Строит столбчатую диаграмму средней стоимости с погрешностями (отклонениями).
    """
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, summary['avg_cost'], yerr=summary['deviation'], capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(summary['method'])
    ax.set_ylabel('Average Best Cost')
    ax.set_title('Methods Comparison: Avg Cost with Deviation')
    plt.tight_layout()
    plt.show()


def plot_boxplot(df):
    """
    Строит boxplot распределения best_cost для каждого метода.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [df[df['method']==m]['best_cost'] for m in df['method'].unique()]
    ax.boxplot(data, labels=df['method'].unique())
    ax.set_ylabel('Best Cost')
    ax.set_title('Distribution of Best Cost by Method')
    plt.tight_layout()
    plt.show()


def pairwise_tests(df):
    """
    Выполняет парные U-тесты Манна-Уитни между методами и выводит матрицу p-значений.
    """
    methods = df['method'].unique()
    pvals = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                data1 = df[df['method']==m1]['best_cost']
                data2 = df[df['method']==m2]['best_cost']
                stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
                pvals.loc[m1, m2] = p
                pvals.loc[m2, m1] = p
            elif i == j:
                pvals.loc[m1, m2] = np.nan
    print("Pairwise Mann-Whitney U test p-values:")
    print(pvals)
    return pvals


def plot_rank(df):
    """
    Ранжирует методы по cost внутри каждой задачи, вычисляет средний ранг и строит bar plot.
    """
    pivot = df.pivot_table(index=['task_id', 'method'], values='best_cost', aggfunc='mean').reset_index()
    pivot['rank'] = pivot.groupby('task_id')['best_cost'].rank(method='min')
    avg_rank = pivot.groupby('method')['rank'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(avg_rank.index, avg_rank.values)
    ax.set_ylabel('Average Rank')
    ax.set_title('Average Rank of Methods Across Tasks')
    plt.tight_layout()
    plt.show()


def plot_correlation(df):
    """
    Строит корреляционную матрицу (heatmap) стоимости лучших решений между методами.
    """
    # Формируем матрицу задач x методов
    pivot = df.pivot_table(index='task_id', columns='method', values='best_cost', aggfunc='mean')
    corr = pivot.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(corr, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title('Correlation Matrix of Methods Best Cost', pad=20)
    plt.tight_layout()
    plt.show()


def plot_radar(summary):
    """
    Строит radar chart по avg_cost и avg_time для каждого метода.
    """
    metrics = ['avg_cost', 'avg_time']
    num_metrics = len(metrics)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)

    for _, row in summary.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row['method'])
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('Radar Chart: Methods Performance')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    summary = summarize(df)
    print("Summary by method:")
    print(summary)
    plot_bar(summary)
    plot_boxplot(df)
    pairwise_tests(df)
    plot_rank(df)
    plot_correlation(df)
    plot_radar(summary)

if __name__ == '__main__':
    main()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Загрузим данные
# df = pd.read_excel("test_results_comparison.xlsx")

# # Убедимся, что названия столбцов правильные
# # assert all(col in df.columns for col in ["task_id", "run_id", "best_cost", "time_sec"])

# # Установим стиль
# sns.set(style="whitegrid")

# # 🎯 График сравнения затрат
# plt.figure(figsize=(12, 6))
# sns.barplot(data=df, y="Затраты", hue="Тип метода")
# plt.title("Сравнение затрат по задачам")
# plt.ylabel("Суммарные затраты")
# plt.xticks(rotation=45)
# plt.legend(title="Метод", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# # plt.savefig("comparison_costs.png")
# # plt.savefig("multiindex_1_results.png")
# plt.show()

# # ⏱️ График сравнения времени
# plt.figure(figsize=(12, 6))
# sns.barplot(data=df, x="Задача", y="Время (с)", hue="Тип метода")
# plt.title("Сравнение времени выполнения по задачам")
# plt.ylabel("Время (с)")
# plt.xticks(rotation=45)
# plt.legend(title="Метод", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# # plt.savefig("comparison_time.png")
# plt.show()
