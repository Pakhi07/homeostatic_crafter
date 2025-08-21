import argparse
import homeostatic_crafter
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import numpy as np
from collections import defaultdict
import os
import torch
import matplotlib.pyplot as plt

class AnalysisCallback(BaseCallback):
    def __init__(self, log_interval=4096, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.visited_states = defaultdict(int)
        self.actions = []
        self.positions = []
        self.rewards = []
        self.healths = []
        self.place_stone_actions = 0
        self.zombies_defeated = 0
        self.skeletons_defeated = 0
        self.wake_ups = 0
        self.death_count = 0
        self.achievements_unlocked = defaultdict(int)
        self.episodes = 0

    def _on_step(self):
        # Access info from the environment
        info = self.locals['infos'][-1]
        action = self.locals['actions'][-1]
        action_name = self.training_env.get_attr('action_names')[0][action]
        
        if action_name == 'place_stone':
            self.place_stone_actions += 1
        
        if info.get('discount') == 0.0:
            self.death_count += 1

        # Check for newly unlocked achievements in this step
        current_achievements = info.get('achievements', {})
        if current_achievements.get('defeat_zombie', 0) > self.achievements_unlocked.get('defeat_zombie', 0):
            self.zombies_defeated += 1
        
        if current_achievements.get('defeat_skeleton', 0) > self.achievements_unlocked.get('defeat_skeleton', 0):
            self.skeletons_defeated += 1

        if current_achievements.get('wake_up', 0) > self.achievements_unlocked.get('wake_up', 0):
            self.wake_ups += 1
            
        # Update our running count of achievements for the next step (only one line needed)
        self.achievements_unlocked.update(current_achievements)

        self.rewards.append(info.get('reward', 0))
        self.actions.append(action) # Use the action variable we already have
        
        obs = self.locals['new_obs']['obs'][0]
        state_key = str(obs.tobytes())
        self.visited_states[state_key] += 1
        
        self.healths.append(info.get('player_health', 0))
        self.positions.append(info.get('player_pos', (0, 0)))
        if 'episodes' in info:
            self.episodes = max(self.episodes, info['episodes'])  
        
        if self.n_calls > 0 and self.n_calls % self.log_interval == 0:
            metrics = self.compute_metrics()
            for key, value in metrics.items():
                self.logger.record(f'custom/{key}', value)
            print(f"Step {self.n_calls}: Logged metrics to TensorBoard.")
            
            self.rewards.clear()
            self.healths.clear()
            self.positions.clear()
            self.actions.clear()
            torch.cuda.empty_cache()

        return True

    def compute_metrics(self):
        positions = np.array(self.positions)
        exploration_variance = np.var(positions, axis=0).sum() if len(positions) > 1 else 0
        state_counts = np.array(list(self.visited_states.values()))
        state_probs = state_counts / (state_counts.sum() + 1e-10)
        state_entropy = -np.sum(state_probs * np.log2(state_probs + 1e-10))
        action_counts = np.bincount(self.actions, minlength=self.training_env.get_attr('action_space')[0].n)
        action_probs = action_counts / (action_counts.sum() + 1e-10)
        action_entropy = -np.sum(action_probs * np.log2(action_probs + 1e-10))
        reward_mean = np.mean(self.rewards)
        health_mean = np.mean(self.healths) if self.healths else 0
        return {
            'exploration_variance': exploration_variance,
            'state_entropy': state_entropy,
            'action_entropy': action_entropy,
            'reward_homeostatic_mean': reward_mean,
            'health_mean': health_mean,
            'total_zombie_defeated': self.zombies_defeated,
            'total_skeleton_defeated': self.skeletons_defeated,
            'total_wake_ups': self.wake_ups,
            'total_stones_placed': self.place_stone_actions,
            'total_deaths': self.death_count,
            'total_episodes': self.episodes,
        }

def save_plots(metrics_dict, save_dir, env_name, seed):
    os.makedirs(save_dir, exist_ok=True)
    
    # Bar plot for scalar metrics
    scalar_metrics = {k: v for k, v in metrics_dict.items() if np.isscalar(v)}
    keys = list(scalar_metrics.keys())
    values = list(scalar_metrics.values())
    
    plt.figure(figsize=(10, 6))
    plt.barh(keys, values, color='skyblue')
    plt.xlabel('Value')
    plt.title(f'{env_name} Final Metrics (Seed {seed})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{env_name}_seed{seed}_metrics_bar.png"))
    plt.close()
    
    print(f"Saved metrics plot to {save_dir}")


def run_training(args, hybrid_lambda):
    assert 0.0 <= hybrid_lambda <= 1.0, "hybrid_lambda must be in [0,1]"

    np.random.seed(args.seed)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=args.outdir,
        name_prefix=f"{args.env}_lambda{hybrid_lambda}_seed{args.seed}_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    env_class = homeostatic_crafter.Env
    env = env_class(seed=args.seed, hybrid_lambda=hybrid_lambda)
    env = homeostatic_crafter.Recorder(
        env, 
        f"{args.outdir}/{args.env}_lambda{hybrid_lambda}_eval/seed_{args.seed}",
        save_stats=True,
        save_episode=False,
        save_video=False,
    )

    env = DummyVecEnv([lambda: env])

    model = stable_baselines3.PPO(
        'MultiInputPolicy', 
        env, 
        verbose=1, 
        tensorboard_log=args.outdir,
        seed=args.seed
    )

    analysis_callback = AnalysisCallback(log_interval=4096)
    callback = CallbackList([analysis_callback, checkpoint_callback])
    
    print(f"Starting training: lambda={hybrid_lambda}, seed={args.seed}")  
    
    model.learn(total_timesteps=int(args.steps), callback=callback)
    
    metrics = analysis_callback.compute_metrics()
    print(f"Final metrics for lambda={hybrid_lambda}: {metrics}")
    try:
        final_log_path = model.logger.dir
        os.makedirs(final_log_path, exist_ok=True)
        
        with open(f"{final_log_path}/{args.env}_seed{args.seed}_metrics.txt", 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
                
        save_plots(metrics, final_log_path, args.env, args.seed)
        model.save(f"{final_log_path}/{args.env}_seed{args.seed}_model")
        
        print(f"Metrics, plots, and model saved successfully to {final_log_path}")
    except Exception as e:
        print(f"Warning: Could not save to {args.outdir}: {e}")

    return metrics




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='homeostatic', choices=['crafter', 'homeostatic'])
    parser.add_argument('--outdir', type=str, default='logdir/lambda_sweep')
    parser.add_argument('--steps', type=float, default=100000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    lambda_values = [0.75, 1.0]  # Sweep from 0 to 1 in 0.2 steps
    results = {}

    for lam in lambda_values:
        metrics = run_training(args, lam)
        results[lam] = metrics

    # Plotting example for reward vs lambda
    # rewards = [results[lam]['reward_homeostatic_mean'] for lam in lambda_values]

    # plt.figure(figsize=(6,4))
    # plt.plot(lambda_values, rewards, marker='o')
    # plt.xlabel("Hybrid Lambda")
    # plt.ylabel("Mean Homeostatic Reward")
    # plt.title("Lambda Sweep Performance")
    # plt.grid(True)
    # plt.savefig(f"{args.outdir}/lambda_sweep_reward.png")
    # plt.show()

if __name__ == '__main__':
    main()
