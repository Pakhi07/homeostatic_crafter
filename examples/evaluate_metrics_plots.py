import argparse
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
# All your plotting functions (plot_lambda_distributions, aggregate_for_plotting, 
# plot_lambda_bar_charts, plot_lambda_line_plot) go here, unchanged.

def plot_lambda_distributions(results, metric_key, outdir, model_order, bins_cache):
    # ... (code is identical to your original script) ...
    if metric_key == 'episode':
        return
    plt.figure(figsize=(10, 6))
    colors = list(mcolors.TABLEAU_COLORS)[:len(model_order)]
    def extract_alpha(name):
        match = re.search(r'_(\d+p\d+|\d+)_model', name)
        if match:
            alpha_str = match.group(1)
            return float(alpha_str.replace('p', '.')) if 'p' in alpha_str else float(alpha_str)
        return 0.0
    sorted_models = sorted(model_order, key=extract_alpha)
    bins = bins_cache.get(metric_key)
    if bins is None:
        all_values = []
        for model_name in sorted_models:
            episode_metrics = results[model_name]
            values = [m[metric_key] for m in episode_metrics if metric_key in m]
            all_values.extend(values)
        if not all_values:
            print(f"No data for {metric_key}, skipping combined plot.")
            return
        global_min = min(all_values)
        global_max = max(all_values)
        if global_min == global_max:
            print(f"All values identical for {metric_key}, skipping combined plot.")
            return
        bins = np.linspace(global_min, global_max, 21)
        bins_cache[metric_key] = bins
    for idx, model_name in enumerate(sorted_models):
        episode_metrics = results[model_name]
        values = [m[metric_key] for m in episode_metrics if metric_key in m]
        if values:
            plt.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.5,
                label=f'α={extract_alpha(model_name)}',
                color=colors[idx % len(colors)]
            )
    plt.xlabel(metric_key.replace('_', ' ').title())
    plt.ylabel('Probability Density')
    plt.title(f'Distribution of {metric_key.replace("_", " ").title()} across Episodes')
    plt.legend(title='Alpha Values', loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'lambda_distribution_{metric_key}.png'), dpi=300)
    plt.close()
    print(f"Saved combined distribution plot for {metric_key}")

def aggregate_for_plotting(results, metric_keys):
    # ... (code is identical to your original script) ...
    summary_data = []
    def extract_alpha(name):
        match = re.search(r'_(\d+p\d+|\d+)_model', name)
        if match:
            alpha_str = match.group(1).replace('p', '.')
            return float(alpha_str)
        return 0.0
    for model_name, episodes in results.items():
        alpha = extract_alpha(model_name)
        for key in metric_keys:
            values = [e.get(key, 0) for e in episodes]
            if values:
                summary_data.append({
                    'alpha': alpha,
                    'metric': key,
                    'mean': np.mean(values),
                    'std': np.std(values)
                })
    return pd.DataFrame(summary_data)

def plot_lambda_bar_charts(summary_df, metric_keys, outdir):
    # ... (code is identical to your original script) ...
    for key in metric_keys:
        metric_df = summary_df[summary_df['metric'] == key]
        if metric_df.empty:
            continue
        plt.figure(figsize=(10, 6))
        plt.bar(
            metric_df['alpha'],
            metric_df['mean'],
            yerr=metric_df['std'],
            capsize=5,
            color=plt.cm.viridis(np.linspace(0, 1, len(metric_df)))
        )
        plt.xlabel('Alpha (λ) Value')
        plt.ylabel(key.replace('_', ' ').title())
        plt.title(f'Mean {key.replace("_", " ").title()} vs. Alpha')
        plt.xticks(metric_df['alpha'])
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'lambda_barchart_{key}.png'), dpi=300)
        plt.close()
        print(f"Saved bar chart for {key}")

def plot_lambda_line_plot(summary_df, metric_key, outdir):
    # ... (code is identical to your original script) ...
    metric_df = summary_df[summary_df['metric'] == metric_key].sort_values('alpha')
    if metric_df.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(metric_df['alpha'], metric_df['mean'], marker='o', linestyle='-')
    plt.fill_between(
        metric_df['alpha'],
        metric_df['mean'] - metric_df['std'],
        metric_df['mean'] + metric_df['std'],
        alpha=0.2
    )
    plt.xlabel('Alpha (λ) Value')
    plt.ylabel(metric_key.replace('_', ' ').title())
    plt.title(f'Performance Trade-off: {metric_key.replace("_", " ").title()} vs. Alpha')
    plt.xticks(metric_df['alpha'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'lambda_lineplot_{metric_key}.png'), dpi=300)
    plt.close()
    print(f"Saved line plot for {metric_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Directory containing the .jsonl metric files')
    parser.add_argument('--outdir', type=str, required=True, help='Directory to save the plots')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Load all .jsonl files from the data directory ---
    all_results = {}
    model_order = []
    for filepath in glob.glob(os.path.join(args.datadir, '*_episode_metrics.jsonl')):
        model_name = os.path.splitext(os.path.basename(filepath))[0].replace('_episode_metrics', '')
        model_order.append(model_name)
        with open(filepath, 'r') as f:
            all_results[model_name] = [json.loads(line) for line in f]
    
    print(f"Loaded data for models: {list(all_results.keys())}")

    all_metrics = set()
    for episode_metrics in all_results.values():
        for m in episode_metrics:
            all_metrics.update(m.keys())
    all_metrics.discard('episode')

    # --- Generate all plots from the loaded data ---
    summary_data = aggregate_for_plotting(all_results, all_metrics)
    plot_lambda_bar_charts(summary_data, all_metrics, args.outdir)
    plot_lambda_line_plot(summary_data, 'total_steps', args.outdir)
    bins_cache = {}
    for metric_key in sorted(all_metrics):
        plot_lambda_distributions(all_results, metric_key, args.outdir, model_order, bins_cache)
    
    print("\nPlot generation complete.")