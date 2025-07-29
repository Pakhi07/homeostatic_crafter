import argparse
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
    parser.add_argument('--max_steps', type=int, default=2000, help='Max steps per evaluation episode.')
    args = parser.parse_args()

    # --- 1. Set up the correct environment ---
    print("Setting up Homeostatic Crafter for evaluation.")
    env = homeostatic_crafter.Env()
    
    # --- 2. Add the necessary wrappers ---
    # The Recorder will save the achievements to stats.jsonl
    env = homeostatic_crafter.Recorder(
        env,
        args.outdir,
        save_stats=True,
        save_video=True, # Set to True if you want videos
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

    while episodes_ran < args.episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        steps_this_episode += 1

        # Check for timeout in addition to the done flag
        timed_out = steps_this_episode >= args.max_steps

        if dones[0] or timed_out:
            episodes_ran += 1
            print(f"Episode {episodes_ran}/{args.episodes} finished (Reason: {'Timeout' if timed_out else 'Done'}).")

            if timed_out:
                print("time out")
                obs = env.reset()
            
            steps_this_episode = 0 # Reset step counter for next episode
            
    env.close()
    print(f"\nEvaluation finished. Achievement stats saved in {args.outdir}/stats.jsonl")

if __name__ == '__main__':
    main()