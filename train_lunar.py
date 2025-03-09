import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create a custom reward function to penalize excessive fuel consumption
def reward_wrapper(env):
    class CustomRewardEnv(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
        
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Penalize fuel usage (thruster activations: actions 1, 2, and 3)
            if action in [1, 2, 3]:
                reward -= 0.5  # Adjust penalty strength as needed
            
            return obs, reward, terminated, truncated, info
    
    return CustomRewardEnv(env)

# Create the LunarLander environment and apply the reward wrapper
env = gym.make("LunarLander-v3")
env = reward_wrapper(env)

# Vectorize the environment for stable training
env = DummyVecEnv([lambda: env])

# Create the PPO agent with optimized hyperparameters
model = PPO(
    "MlpPolicy", 
    env,
    learning_rate=0.0002,  # Slightly increased learning rate
    gamma=0.99,  # Discount factor for long-term rewards
    n_steps=2048,  # Number of steps per update
    batch_size=64,  # Mini-batch size
    n_epochs=10,  # Number of training epochs per update
    clip_range=0.2,  # Clipping for PPO
    ent_coef=0.01,  # Entropy coefficient to encourage exploration
    verbose=1
)

# Train the agent for n timesteps
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("ppo_lunar_lander")
