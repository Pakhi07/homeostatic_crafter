import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import homeostatic_crafter
import re

def compute_metrics(actions, positions, rewards, healths, visited_states, action_names, achievements, death_count):
    metrics = {}
    metrics['total_steps'] = len(actions)
    metrics['unique_states'] = len(visited_states)
    metrics['exploration_variance'] = np.var(list(visited_states.values())) if visited_states else 0
    metrics['action_entropy'] = (
        -np.sum([p * np.log(p) for p in np.bincount(actions, minlength=len(action_names)) / max(len(actions), 1) if p > 0])
    )
    metrics['reward_extrinsic_mean'] = np.mean(rewards) if rewards else 0
    metrics['reward_extrinsic_sum'] = np.sum(rewards) if rewards else 0
    metrics['health_mean'] = np.mean(healths) if healths else 0
    metrics['deaths'] = death_count
    metrics.update(achievements)
    return metrics

def evaluate_model(model_path, args):
    print(f"\n=== Evaluating model: {model_path} ===")

    model_name = os.path.splitext(os.path.basename(model_path))[0]  # e.g., homeostatic_seed0_0p25_model
    model_outdir = os.path.join(args.outdir, model_name)  # e.g., logdir/evaluation/.../homeostatic_seed0_0p25_model
    os.makedirs(model_outdir, exist_ok=True)


    env = homeostatic_crafter.Env()
    env = homeostatic_crafter.Recorder(
        env,
        model_outdir,
        save_stats=True,
        save_video=True,
        save_episode=False
    )

    env = DummyVecEnv([lambda: env])
    model = PPO.load(model_path)

    visited_states = defaultdict(int)
    actions, positions, rewards, healths = [], [], [], []
    death_count = 0
    episodes_ran = 0
    steps_this_episode = 0
    obs = env.reset()
    action_names = env.get_attr('action_names')[0]
    per_episode_metrics = []

    while episodes_ran < args.episodes:
        achievements = defaultdict(int)  # Reset achievements per episode
        death_count = 0  # Reset death_count per episode
        done = [False]
        steps_this_episode = 0
        visited_states = defaultdict(int)
        actions, positions, rewards, healths = [], [], [], []

        while not done[0] and steps_this_episode < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            info = info[0]
            steps_this_episode += 1

            actions.append(action[0])
            rewards.append(reward[0])
            healths.append(info.get('player_health', 0))
            positions.append(info.get('player_pos', (0, 0)))

            state_key = str(obs["obs"].tobytes())
            visited_states[state_key] += 1

            if action_names[action[0]] == 'place_stone':
                achievements['place_stone'] += 1

            if info.get('discount') == 0.0:
                death_count += 1

            for key in ['defeat_zombie', 'defeat_skeleton', 'wake_up']:
                if info.get('achievements', {}).get(key, 0) > achievements.get(key, 0):
                    achievements[key] += 1

        episodes_ran += 1
        episode_metrics = compute_metrics(
            actions, positions, rewards, healths, visited_states, action_names, achievements, death_count
        )
        episode_metrics['episode'] = episodes_ran
        per_episode_metrics.append(episode_metrics)

        obs = env.reset() if episodes_ran < args.episodes else None

    env.close()

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    with open(os.path.join(args.outdir, f'{model_name}_episode_metrics.jsonl'), 'w') as f:
        for m in per_episode_metrics:
            m_clean = {k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v) for k, v in m.items()}
            json.dump(m_clean, f)
            f.write('\n')

    print(f"Saved metrics for {model_name}")
    return model_name, per_episode_metrics

def plot_lambda_distributions(results, metric_key, outdir, model_order, bins_cache):
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

def plot_lambda_separate_distributions(results, metric_key, outdir, model_order, bins_cache):
    if metric_key == 'episode':
        return

    colors = list(mcolors.TABLEAU_COLORS)[:len(model_order)]

    def extract_alpha(name):
        match = re.search(r'_(\d+p\d+|\d+)_model', name)
        if match:
            alpha_str = match.group(1)
            return float(alpha_str.replace('p', '.')) if 'p' in alpha_str else float(alpha_str)
        return 0.0

    sorted_models = sorted(model_order, key=extract_alpha)
    bins = bins_cache.get(metric_key)

    for idx, model_name in enumerate(sorted_models):
        episode_metrics = results[model_name]
        values = [m[metric_key] for m in episode_metrics if metric_key in m]
        if not values:
            print(f"No data for {metric_key} in {model_name}, skipping separate plot.")
            continue
        if min(values) == max(values):
            print(f"All values identical for {metric_key} in {model_name}, skipping separate plot.")
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.7,
            color=colors[idx % len(colors)]
        )
        plt.xlabel(metric_key.replace('_', ' ').title())
        plt.ylabel('Probability Density')
        plt.title(f'Distribution of {metric_key.replace("_", " ").title()} (α={extract_alpha(model_name)})')
        plt.grid(True)
        plt.tight_layout()
        alpha_str = str(extract_alpha(model_name)).replace('.', 'p')
        plt.savefig(os.path.join(outdir, f'lambda_distribution_{metric_key}_alpha_{alpha_str}.png'), dpi=300)
        plt.close()
        print(f"Saved separate distribution plot for {metric_key}, alpha={extract_alpha(model_name)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, help='List of PPO model .zip paths')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes per model')
    parser.add_argument('--max_steps', type=int, default=2000, help='Max steps per episode')
    parser.add_argument('--outdir', type=str, required=True, help='Directory to save results')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    all_results = {}
    model_order = []
    for model_path in args.model_paths:
        model_name, per_episode_metrics = evaluate_model(model_path, args)
        all_results[model_name] = per_episode_metrics
        model_order.append(model_name)

    print("Processed models:", model_order)
    print("Unique results keys:", list(all_results.keys()))

    all_metrics = set()
    for episode_metrics in all_results.values():
        for m in episode_metrics:
            all_metrics.update(m.keys())
    all_metrics.discard('episode')

    bins_cache = {}
    for metric_key in sorted(all_metrics):
        plot_lambda_distributions(all_results, metric_key, args.outdir, model_order, bins_cache)
        plot_lambda_separate_distributions(all_results, metric_key, args.outdir, model_order, bins_cache)