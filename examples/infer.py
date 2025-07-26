# enjoy.py

import argparse
# import crafter
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import homeostatic_crafter
from CompatibilityWrapper import CompatibilityWrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved .zip model file.')
    parser.add_argument('--env', type=str, default='homeostatic', choices=['crafter', 'homeostatic'])
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run for evaluation.')
    parser.add_argument('--outdir', type=str, default='logdir/evaluation', help='Directory to save evaluation stats.')
    args = parser.parse_args()

    # --- 1. Set up the correct environment ---
    if args.env == 'homeostatic':
        print("Setting up Homeostatic Crafter for evaluation.")
        env = homeostatic_crafter.Env()
        # env = CompatibilityWrapper(env)
    else:
        print("Setting up standard Crafter for evaluation.")
        # env = crafter.Env()
        
    # --- 2. Add the necessary wrappers ---
    # The Recorder will save the achievements to stats.jsonl
    env = homeostatic_crafter.Recorder(
        env,
        args.outdir,
        save_stats=True,
        save_video=False, # Set to True if you want videos
        save_episode=False
    )
    env = CompatibilityWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    # --- 3. Load the pre-trained model ---
    print(f"Loading model from: {args.model_path}")
    model = PPO.load(args.model_path)
    
    # --- 4. Run the evaluation loop ---
    obs = env.reset()
    episodes_ran = 0

    steps_this_episode = 0
    MAX_STEPS_PER_EVAL_EPISODE = 2000

    while episodes_ran < args.episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        steps_this_episode += 1

        # Check for timeout in addition to the done flag
        if dones[0] or steps_this_episode >= MAX_STEPS_PER_EVAL_EPISODE:
            episodes_ran += 1
            steps_this_episode = 0 # Reset step counter for next episode
            print(f"Episode {episodes_ran}/{args.episodes} finished.")
            
    env.close()
    print(f"\nEvaluation finished. Achievement stats saved in {args.outdir}/stats.jsonl")

if __name__ == '__main__':
    main()