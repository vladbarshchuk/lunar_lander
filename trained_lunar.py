import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

LOG_FILE = "lunar_lander_log.txt"

# Create and vectorize the LunarLander environment with rendering enabled.
env = gym.make("LunarLander-v3", render_mode="human")
env = DummyVecEnv([lambda: env])

# Load the trained model.
model = PPO.load("ppo_lunar_lander", env=env)

# Reset the environment.
reset_result = env.reset()
if isinstance(reset_result, tuple):
    obs, info = reset_result
else:
    obs = reset_result

done = False
episode_reward = 0
fuel_consumed = 0  # Track fuel consumption

while not done:
    action, _ = model.predict(obs)
    result = env.step(action)

    # Check if the returned result has 4 or 5 elements.
    if len(result) == 4:
        obs, reward, done, info = result
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        raise ValueError("Unexpected number of values returned from env.step()")

    # Since you're using DummyVecEnv, reward is a list
    episode_reward += reward[0]

    # Estimate fuel consumption: count thruster actions (actions 1, 2, and 3)
    if action in [1, 2, 3]:
        fuel_consumed += 1

    env.render()

# Print outcome after episode ends.
if episode_reward >= 200:
    print("Congratulations! Successful landing!")
else:
    print("Landing was not successful. Try again!")

print(f"Landing Score: {episode_reward}, Fuel Consumed: {fuel_consumed}")

# Append results to log file
with open(LOG_FILE, "a") as log:
    log.write(f"Score: {episode_reward}, Fuel Consumed: {fuel_consumed}\n")

env.close()
