import argparse
import numpy as np
import pandas as pd
import os
import random
import warnings

import torch

import ray
from ray import tune, serve
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from envs import SingleSatEnv
from help_functions import data_preprocess

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Set the response deadline for the queue length manually to avoid the "Queue length exceeded" error
# This is a workaround for the issue in Ray 2.6.0
# You can adjust the value as needed based on your system's capabilities
RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S = float(
    os.environ.get("RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S", 0.5)
)

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.filterwarnings(
    "ignore",
    message=r".*RLModule\(config=\[RLModuleConfig object\]\).*deprecated.*",
    category=DeprecationWarning,
)

# Initialize Ray
if ray.is_initialized():
    try:
        serve.status()  # Check if Ray Serve is already running
        serve.shutdown()  # Shutdown Ray Serve if it's running
    except Exception as e:
        print(f"Ray Serve shutdown failed: {e}")
    ray.shutdown()  # Shutdown Ray if it's initialized

ray.init(
    ignore_reinit_error=True,
    log_to_driver=True,
    logging_level="INFO",
)  # Initialize Ray with driver logs enabled for better Tune visibility

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
    "render_mode": None, 
    }

register_env("single_sat_env", lambda config: SingleSatEnv(env_config))

parser = argparse.ArgumentParser(description="Train a PPO agent in the SingleSatEnv environment.")
parser.add_argument("--num-cpus", type=int, default=4, help="Number of CPUs to use for training.")
parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use for training.")
parser.add_argument("--num-learners", type=int, default=1, help="Number of workers for parallel training.")
parser.add_argument("--num-cpus-per-learner", type=int, default=2, help="Number of CPUs per learner.")
parser.add_argument("--num-gpus-per-learner", type=int, default=1, help="Number of GPUs per learner.")
parser.add_argument("--tune-verbose", type=int, default=2, choices=[0, 1, 2, 3], help="Tune verbosity level.")
args = parser.parse_args()

training_iterations = 50
episode_length = 1440 
num_envs_per_env_runner = 4  # Number of environments per environment runner

config = (
    PPOConfig()
    .environment(env="single_sat_env", env_config=env_config)
    .framework("torch")
    .fault_tolerance(restart_failed_sub_environments=True)
    .learners(
        num_learners=args.num_learners,
        num_cpus_per_learner=args.num_cpus_per_learner,
        num_gpus_per_learner=args.num_gpus_per_learner
    )
    .env_runners(
        num_env_runners=1,
        num_envs_per_env_runner=num_envs_per_env_runner,
        rollout_fragment_length="auto", # change to auto if needed
        batch_mode="complete_episodes",
    )
    .reporting(metrics_num_episodes_for_smoothing=10)
    .training(lr=1e-4, train_batch_size_per_learner=episode_length * num_envs_per_env_runner)
    .resources(num_gpus=args.num_gpus)
    .evaluation(
        evaluation_interval=5,
        evaluation_duration=10,
        evaluation_duration_unit="episodes",
        evaluation_parallel_to_training=False,
    )
)

reporter = tune.CLIReporter(
    metric_columns=[
        "training_iteration",
        "episode_reward_mean",
        "timesteps_total",
        "time_total_s",
    ]
)

# Create a Tuner instance
tuner = tune.Tuner(
    "PPO",
    param_space=config,
    tune_config=tune.TuneConfig(
        num_samples=1,
        max_concurrent_trials=1,
        reuse_actors=False,
    ),
    run_config=tune.RunConfig(
        name="ppo_single_sat_env",
        stop={"training_iteration": training_iterations},
        verbose=args.tune_verbose,
        progress_reporter=reporter,
        log_to_file=True,
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=5,
            checkpoint_at_end=True,
        ),
    ),
)

# Fit the tuner to start training
results = tuner.fit()

ray.shutdown()  # Shutdown Ray if it's initialized