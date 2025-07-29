import argparse
from stable_baselines3 import PPO
import homeostatic_crafter
from CompatibilityWrapper import CompatibilityWrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved .zip model file.')
    parser.add_argument('--env', type=str, default='homeostatic', choices=['crafter', 'homeostatic'])
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run for evaluation.')
    parser.add_argument('--outdir', type=str, default='logdir/evaluation', help='Directory to save evaluation stats.')
    parser.add_argument('--max_steps', type=int, default=2000, help='Max steps per evaluation episode.')
    args = parser.parse_args()

    # --- 1. Set up the environment ---
    env = homeostatic_crafter.Env()
    env = homeostatic_crafter.Recorder(
        env,
        args.outdir,
        save_stats=True,
        save_video=False,
        save_episode=False
    )
    env = CompatibilityWrapper(env)

    # --- 2. Load the model ---
    print(f"Loading model from: {args.model_path}")
    model = PPO.load(args.model_path)

    # --- 3. Evaluation loop ---
    episodes_ran = 0
    obs = env.reset()
    steps_this_episode = 0

    while episodes_ran < args.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        steps_this_episode += 1

        timed_out = steps_this_episode >= args.max_steps
        if done or timed_out:
            episodes_ran += 1
            print(f"Episode {episodes_ran}/{args.episodes} finished (Reason: {'Timeout' if timed_out else 'Done'}).")
            obs = env.reset()
            steps_this_episode = 0

    env.close()
    print(f"\nEvaluation finished. Achievement stats saved in {args.outdir}/stats.jsonl")

if __name__ == '__main__':
    main()