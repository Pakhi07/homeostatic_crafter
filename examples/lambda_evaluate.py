import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
import os

def load_data(json_path):
    data_by_alpha = defaultdict(list)
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    episodes_per_alpha = 150

    with open(json_path, 'r') as f:
        all_data = [json.loads(line) for line in f]

    if len(all_data) != len(alpha_values) * episodes_per_alpha:
        print(f"Warning: Expected {len(alpha_values) * episodes_per_alpha} rows, got {len(all_data)}")

    for i, episode_data in enumerate(all_data):
        alpha = alpha_values[i // episodes_per_alpha]
        episode_data['alpha'] = alpha
        data_by_alpha[alpha].append(episode_data)

    alpha_order = sorted(alpha_values)
    return data_by_alpha, alpha_order

def plot_achievement_distributions(data_by_alpha, metric_key, outdir, alpha_order, bins_cache):
    plt.figure(figsize=(10, 6))
    colors = list(mcolors.TABLEAU_COLORS)[:len(alpha_order)]

    all_values = []
    for alpha in alpha_order:
        values = [ep[metric_key] for ep in data_by_alpha[alpha] if metric_key in ep]
        all_values.extend(values)
    
    if not all_values:
        print(f"No data for {metric_key}, skipping combined plot.")
        return
    
    max_value = max(all_values)
    bins = np.arange(-0.5, max_value + 1.5, 1)  # Center bins on integers
    bins_cache[metric_key] = bins

    for idx, alpha in enumerate(alpha_order):
        values = [ep[metric_key] for ep in data_by_alpha[alpha] if metric_key in ep]
        if values:
            plt.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.5,
                label=f'α={alpha}',
                color=colors[idx % len(colors)]
            )

    plt.xlabel(metric_key.replace('achievement_', '').replace('_', ' ').title())
    plt.ylabel('Probability Density')
    plt.title(f'Distribution of {metric_key.replace("achievement_", "").replace("_", " ").title()} across Episodes')
    plt.legend(title='Alpha Values', loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'achievement_distribution_{metric_key}.png'), dpi=300)
    plt.close()
    print(f"Saved combined distribution plot for {metric_key}")

def plot_achievement_separate_distributions(data_by_alpha, metric_key, outdir, alpha_order, bins_cache):
    colors = list(mcolors.TABLEAU_COLORS)[:len(alpha_order)]
    bins = bins_cache.get(metric_key)

    for idx, alpha in enumerate(alpha_order):
        values = [ep[metric_key] for ep in data_by_alpha[alpha] if metric_key in ep]
        if not values:
            print(f"No data for {metric_key} in α={alpha}, skipping separate plot.")
            continue
        if min(values) == max(values):
            print(f"All values identical for {metric_key} in α={alpha}, skipping separate plot.")
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.7,
            color=colors[idx % len(colors)]
        )
        plt.xlabel(metric_key.replace('achievement_', '').replace('_', ' ').title())
        plt.ylabel('Probability Density')
        plt.title(f'Distribution of {metric_key.replace("achievement_", "").replace("_", " ").title()} (α={alpha})')
        plt.grid(True)
        plt.tight_layout()
        alpha_str = str(alpha).replace('.', 'p')
        plt.savefig(os.path.join(outdir, f'achievement_distribution_{metric_key}_alpha_{alpha_str}.png'), dpi=300)
        plt.close()
        print(f"Saved separate distribution plot for {metric_key}, alpha={alpha}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True, help='Path to merged JSONL file')
    parser.add_argument('--outdir', type=str, default='logdir/evaluation/lambda_sweep/Aug14_Final_analysis/', help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data_by_alpha, alpha_order = load_data(args.json_path)

    print("Alpha values processed:", alpha_order)
    print("Episodes per alpha:", {alpha: len(data_by_alpha[alpha]) for alpha in alpha_order})

    achievement_metrics = [
        'achievement_collect_coal', 'achievement_collect_diamond', 'achievement_collect_drink',
        'achievement_collect_iron', 'achievement_collect_sapling', 'achievement_collect_stone',
        'achievement_collect_wood', 'achievement_defeat_skeleton', 'achievement_defeat_zombie',
        'achievement_eat_cow', 'achievement_eat_plant', 'achievement_make_iron_pickaxe',
        'achievement_make_iron_sword', 'achievement_make_stone_pickaxe', 'achievement_make_stone_sword',
        'achievement_make_wood_pickaxe', 'achievement_make_wood_sword', 'achievement_place_furnace',
        'achievement_place_plant', 'achievement_place_stone', 'achievement_place_table',
        'achievement_wake_up'
    ]

    bins_cache = {}
    for metric_key in sorted(achievement_metrics):
        plot_achievement_distributions(data_by_alpha, metric_key, args.outdir, alpha_order, bins_cache)
        plot_achievement_separate_distributions(data_by_alpha, metric_key, args.outdir, alpha_order, bins_cache)