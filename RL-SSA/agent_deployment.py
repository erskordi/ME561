import argparse
from pathlib import Path
import imageio
import numpy as np 
import torch
from ray.rllib.core.rl_module import RLModule
import pandas as pd

import matplotlib.pyplot as plt

from envs import SingleSatEnv
from help_functions import data_preprocess
from policies import no_op_policy, random_policy, max_priority_policy

np.random.seed(42)

parser = argparse.ArgumentParser(description="Run PPO policy deployment rollout.")
parser.add_argument(
    "--stochastic",
    action="store_true",
    help="Sample actions from the policy distribution instead of taking argmax.",
)
args = parser.parse_args()

# Example usage of the IridiumSingleSatEnv
# Load orbital data 
df = pd.read_csv('/home/erskordi/Documents/UNM-files/Summer26/ME561/Project/2026-06-20/ssa_data_audit_outputs/candidate_controlled_iridium_records.csv')
df_clean = data_preprocess(df=df)

# Load other RSOs
df_other = pd.read_csv('/home/erskordi/Documents/UNM-files/Summer26/ME561/Project/2026-06-20/ssa_data_audit_outputs/candidate_background_leo_500_1200_latest.csv')
df_other_clean = df_other[df_other['NORAD_CAT_ID'] != 24946]  # Exclude Iridium 33 from other RSOs
df.drop_duplicates(subset=["NORAD_CAT_ID"], inplace=True, keep='last')
df_other_clean.dropna(subset=["TLE_LINE1", "TLE_LINE2"], inplace=True)
df_other_clean.reset_index(drop=True, inplace=True)
# Sample K unique RSOs from the cleaned other RSOs DataFrame
K = 100  # Number of RSOs to sample
sampled_rso = df_other_clean.sample(n=K)

# Debugging example using the latest available element set for Iridium 33 satellite
iridium33 = df_clean[
    (df_clean['NORAD_CAT_ID'] == 24946) &
    (df_clean['OBJECT_NAME'] == "IRIDIUM 33") &
    (df_clean["OBJECT_TYPE"] == "PAYLOAD")
].sort_values(by="EPOCH").iloc[-1]

orbital_data = {
    'cat_id': iridium33['NORAD_CAT_ID'],
    'epoch': iridium33['EPOCH'],
    'object_name': iridium33['OBJECT_NAME'],
    'object_type': iridium33['OBJECT_TYPE'],
    'tle_line1': iridium33['TLE_LINE1'],
    'tle_line2': iridium33['TLE_LINE2'],
    'a': iridium33['SEMIMAJOR_AXIS'],  # Semi-major axis in km
    'e': iridium33['ECCENTRICITY'],  # Eccentricity
    'i': iridium33['INCLINATION'],  # Inclination in radians
    'raan': iridium33['RA_OF_ASC_NODE'],  # Right ascension of ascending node in radians
    'argp': iridium33['ARG_OF_PERICENTER'],  # Argument of perigee in radians
    'M': iridium33['MEAN_ANOMALY'],  # Mean anomaly in radians
}

env_config = {
    "data": orbital_data,
    "rso_df": sampled_rso,
    "K": 5,
    "rso_pool_size": 200,
    "max_rso_age_days": 30,
    "max_obs_range_km": 1000,
    "range_norm_km": 8000,  # observed candidate ranges reach about 7309 km
    "rel_speed_norm_km_s": 15,
    "risk_distance_scale_km": 2500,
    "render_mode": "rgb_array",  # Set to "rgb_array" for rendering, or None for no rendering
    }

checkpoint_path = Path('/home/erskordi/ray_results/ppo_single_sat_env/PPO_single_sat_env_bfee0_00000_0_2026-07-17_09-16-41/checkpoint_000009')

rl_module = RLModule.from_checkpoint(
    checkpoint_path
    / "learner_group"
    / "learner"
    / "rl_module"
    / "default_policy"
)
env = SingleSatEnv(env_config)

frames = [] # List to store frames for video rendering

episode_return = 0.0
episode_return_noop = 0.0
episode_return_random = 0.0
episode_return_max_priority = 0.0
done = False
obs, info = env.reset(seed=42)
action_counts = np.zeros(env.action_space.n, dtype=int)
frames.append(env.render())  # Capture the initial frame

while not done:
    obs_batch = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
    model_outputs = rl_module.forward_inference({"obs": obs_batch})
    logits = model_outputs["action_dist_inputs"][0]

    if args.stochastic:
        action = int(torch.distributions.Categorical(logits=logits).sample().item())
    else:
        action = int(torch.argmax(logits).item())

    action_counts[action] += 1
    obs, reward, terminated, truncated, info = env.step(action)
    episode_return += reward
    # Evaluate no-op policy
    episode_return_noop += no_op_policy()
    # Evaluate random policy
    episode_return_random += random_policy(env)
    # Evaluate max-priority policy
    episode_return_max_priority += max_priority_policy(env.last_candidates, env)
    frames.append(env.render())  # Capture the frame after taking the action
    done = terminated or truncated

# Save the frames as a GIF
gif_filename = f"single_sat_agent_eval_stochastic_rollout_{args.stochastic}.gif"
imageio.mimsave(gif_filename, frames, fps=2)
print(f"Simulation GIF saved as {gif_filename}")
print(f"Rollout mode: {'stochastic' if args.stochastic else 'deterministic'}")
print(f"Episode return: {episode_return:.3f}")
print(f"Episode return (no-op policy): {episode_return_noop:.3f}")
print(f"Episode return (random policy): {episode_return_random:.3f}")
print(f"Episode return (max-priority policy): {episode_return_max_priority:.3f}")
print("Action counts:", {a: int(c) for a, c in enumerate(action_counts)})

# Create a histogram of action counts
plt.clf()
plt.cla()
plt.bar(range(env.action_space.n), action_counts)
plt.xlabel('Action')
plt.ylabel('Count')
plt.title('Action Counts Histogram')
# Save the histogram as a PNG file
histogram_filename = f"action_counts_histogram_stochastic_rollout_{args.stochastic}.png"
plt.savefig(histogram_filename)
print(f"Action counts histogram saved as {histogram_filename}")
