from calendar import EPOCH

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from sgp4_module import SGP4Propagator
from help_functions import data_preprocess

MU = 398600.4418  # Earth's gravitational parameter in km^3/s^2
R_E = 6378.1  # Earth radius in km

class SingleSatBaseEnv(gym.Env):
    """
    A custom Gymnasium environment for simulating a single Iridium satellite.
    """

    def __init__(self, data):
        super(SingleSatBaseEnv, self).__init__()

        self.orbital_data = data
        self.tle_line_1 = self.orbital_data['tle_line1']
        self.tle_line_2 = self.orbital_data['tle_line2']
        self.current_time = pd.to_datetime(self.orbital_data['epoch'], utc=True)

        self.propagator = SGP4Propagator(self.tle_line_1, self.tle_line_2)

        self.timestep_index = 0
        self.timestep = 60 # timestep in seconds (1 minute)
        self.debug_episode_length = 1 # i.e., one orbit
        self.true_episode_length = 1440 # number of timesteps in an episode (1 day)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(1) # Useful only for debugging
        
        # Observation space could include satellite position, velocity, and other relevant states
        self.observation_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to an initial state and return the initial observation.
        """
        super().reset(seed=seed)
        # Initialize the state 
        self.timestep_index = 0
        self.current_time = pd.to_datetime(self.orbital_data['epoch'], utc=True)
        return self.orbital_data, {}

    def step(self, action):
        """
        Apply the action to the environment and return the new state, reward, done flag, and info.
        """
        # Increment timestep index
        self.timestep_index += 1
        self.current_time += pd.to_timedelta(self.timestep, unit='s')

        # Propagate the satellite's state using SGP4
        propagated = self._propagate(self.current_time)
        obs = self._build_observation(propagated)
        
        # Calculate reward (this is a placeholder for actual reward logic)
        reward = 0
        
        # Check if the episode is done (this is a placeholder for actual termination conditions)
        truncated = terminated = self.timestep_index >= self.debug_episode_length 
        
        info = {
            "time_utc": self.current_time,
            "a_km": obs[0],
            "eccentricity": obs[1],
            "i_rad": obs[2],
            "raan_rad": obs[3],
            "argp_rad": obs[4],
            "true_anomaly_rad": obs[5],
            "altitude_km": propagated[4],
            #"orbit_classification": self.classify_orbit(obs[6])
        }  
        
        return obs, reward, terminated, truncated, info
    
    def _wrap_to_2pi(self, angle_rad):
        """
        Wrap an angle in radians to the range [0, 2π).
        """
        return angle_rad % (2 * np.pi)

    def _build_observation(self, propagated):
        """
        Build the observation from the current state.
        """
        r_vec, v_vec, _, radius, _ = propagated

        # specific angular momentum
        h_vec = np.cross(r_vec, v_vec)
        h_norm = np.linalg.norm(h_vec)

        if h_norm < 1e-8:
            raise ValueError("Specific angular momentum is too small, cannot compute orbital elements.")

        # inclination
        i = np.arccos(h_vec[2] / h_norm)

        # node vector
        k_vec = np.array([0, 0, 1])
        n_vec = np.cross(k_vec, h_vec)
        n = np.linalg.norm(n_vec)

        # eccentricity vector
        e_vec = np.cross(v_vec, h_vec) / MU - r_vec / np.linalg.norm(r_vec)
        e = np.linalg.norm(e_vec)

        # specific orbital energy
        specific_energy = 0.5 * np.linalg.norm(v_vec)**2 - MU / np.linalg.norm(r_vec)
        
        # semi-major axis
        if abs(specific_energy) < 1e-8:
            alpha = np.inf  # Parabolic trajectory
        else:
            alpha = - MU / (2 * specific_energy)

        # right ascension of ascending node
        if n > 1e-8:
            raan = np.arctan2(n_vec[1], n_vec[0])
            raan = self._wrap_to_2pi(raan)
        else:
            raan = 0.0

        # argument of perigee
        if e > 1e-8 and n > 1e-8:
            argp = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
            if e_vec[2] < 0:
                argp = 2 * np.pi - argp
            argp = self._wrap_to_2pi(argp)
        else:
            argp = 0.0

        # true anomaly
        if e > 1e-8:
            true_anomaly = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * radius), -1.0, 1.0))
            if np.dot(r_vec, v_vec) < 0:
                true_anomaly = 2 * np.pi - true_anomaly
            true_anomaly = self._wrap_to_2pi(true_anomaly)
        else:
            # Circular orbit: true anomaly is undefined, but we can set it to 0 for consistency
            if n > 1e-8:
                true_anomaly = np.arccos(
                    np.clip(np.dot(n_vec, r_vec) / (n * radius), -1.0, 1.0)
                )
                if r_vec[2] < 0:
                    true_anomaly = 2 * np.pi - true_anomaly
                true_anomaly = self._wrap_to_2pi(true_anomaly)
            else:
                # Circular equatorial orbit: use true longitude instead
                true_anomaly = np.arctan2(r_vec[1], r_vec[0])
                true_anomaly = self._wrap_to_2pi(true_anomaly)

        # Mean anomaly for elliptical orbit
        if e < 1.0 - 1e-8:  # Ensure it's not a parabolic or hyperbolic orbit
            E = 2 * np.arctan2(
                np.sqrt(1 - e) * np.sin(true_anomaly / 2),
                np.sqrt(1 + e) * np.cos(true_anomaly / 2)
            )
            M = E - e * np.sin(E)
            M = self._wrap_to_2pi(M)
        else:
            M = np.nan  # For parabolic or hyperbolic orbits, mean anomaly is not defined

        normalized_obs = np.array([
            (alpha - 7000) / 1000,
            e,
            i/np.pi,
            raan / (2 * np.pi),
            argp / (2 * np.pi),
            true_anomaly / (2 * np.pi),
            r_vec[0] / 7000,
            r_vec[1] / 7000,
            r_vec[2] / 7000,
            v_vec[0] / 7.5,
            v_vec[1] / 7.5,
            v_vec[2] / 7.5
        ])

        return normalized_obs
    
    def _propagate(self, datetime_utc):
        r_km, v_km_s = self.propagator.propagate(datetime_utc)

        radius_km = np.linalg.norm(r_km)
        velocity_km_s = np.linalg.norm(v_km_s)
        altitude_km = radius_km - R_E

        return r_km, v_km_s, radius_km, velocity_km_s, altitude_km

    def render(self):
        """
        Render the environment (optional).
        """
        pass  # Rendering logic can be implemented here if needed

class SingleSatEnv(SingleSatBaseEnv):
    """
    A custom Gymnasium environment for simulating a single Iridium satellite.
    This class is intended for actual use, while SingleSatTestEnv is for testing and debugging.
    """

    def __init__(self, data, K):
        """
        The state of each candidate RSO will include:
        - relative position (3D)
        - relative velocity (3D)
        - range
        - relative speed
        - visibility status (1 if visible, 0 if not)
        - risk score
        - time since last observed
        - object type (encoded as an integer)
        """
        super(SingleSatEnv, self).__init__(data)
        # K = Number of candidate resident space objects (RSOs) that can be selected for observation in the environment
        
        # Each action corresponds to selecting one of the K RSOs or taking no action (0)
        self.K = K
        self.action_space = spaces.Discrete(self.K + 1)

        # The observation space includes the satellite's state and the states of K RSOs.
        self.observation_space = spaces.Box(low=-1, high=1, shape=(12 + K * 11,), dtype=np.float32)

    def _reward(self, obs, action):
        """
        Calculate the reward based on the current observation and action taken.
        This is a placeholder for actual reward logic.
        """
        reward = 0.0

        if action == 0:
            reward -= 1  # Penalty for taking no action
        else:
            
    

if __name__ == "__main__":
    # Example usage of the IridiumSingleSatEnv
    # Load orbital data 
    df = pd.read_csv('/home/erskordi/Documents/UNM-files/Summer26/ME561/Project/2026-06-20/ssa_data_audit_outputs/candidate_controlled_iridium_records.csv')
    df_clean = data_preprocess(df=df)

    mode = "debug"  # Change to "true" for the actual environment

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
    
    if mode == "debug":
        env = SingleSatBaseEnv(orbital_data)
    else:
        env = SingleSatEnv(orbital_data, K=5)  # Example with K=5
    obs = env.reset()
    print("Initial Observation:", obs)

    for _ in range(10):
        action = env.action_space.sample()  # Sample a random action (for demonstration)
        obs, reward, terminated, truncated, info = env.step(action)
        print("Observation:", obs)
        print("Reward:", reward)
        print("Info:", info)
        if terminated or truncated:
            break
    """"""