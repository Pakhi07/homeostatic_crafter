# examples/wrapper.py
import gym

class CompatibilityWrapper(gym.Wrapper):
    """
    A wrapper to handle the tuple return format of the reset() method,
    making it compatible with Stable-Baselines3's DummyVecEnv.
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        """
        Calls the underlying environment's reset and returns only the observation.
        """
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        """
        Calls the underlying environment's step and returns a 4-tuple,
        as expected by older Gym/SB3 versions.
        """
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, info