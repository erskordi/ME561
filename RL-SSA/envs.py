from calendar import EPOCH

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import imageio.v2 as imageio

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
        terminated = False
        truncated = self.timestep_index >= self.debug_episode_length
        
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

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }

    def __init__(self, 
                 config):
        """
        SingleSatEnv is a custom Gymnasium environment 
        for simulating a single Iridium satellite and its 
        interactions with candidate resident space objects (RSOs).
        The interactions are expressed through the selection of RSOs 
        for observation, and the environment provides feedback in the 
        form of rewards based on the actions taken by the agent.

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
        super(SingleSatEnv, self).__init__(config["data"])
        # K = Number of candidate resident space objects (RSOs) 
        # that can be selected for observation in the environment
        
        # Each action corresponds to selecting one of the K RSOs or 
        # taking no action (0)
        self.config = config
        self.K = config["K"]
        self.rso_feature_dim = 14 # Each RSO has 14 features in the observation space
        self.max_rso_age_days = config["max_rso_age_days"]
        self.rso_pool_size = config["rso_pool_size"]
        self.max_obs_range_km = config["max_obs_range_km"]
        self.range_norm_km = config["range_norm_km"]
        self.rel_speed_norm_km_s = config["rel_speed_norm_km_s"]
        self.risk_distance_scale_km = config["risk_distance_scale_km"]

        self.closing_speed_scale_km_s = 20.0 # km/s

        # reward weights
        self.w_risk = 1.0
        self.w_observability = 0.4
        self.w_staleness = 0.4
        self.w_type = 0.15
        self.w_missed_priority = 0.6
        self.noop_base_penalty = 0.05
        self.noop_priority_penalty = 0.5
        self.retask_penalty = 0.01

        self.max_time_since_obs_s = 24 * 3600  # 1 day in seconds
        self.previous_action = 0  # Initialize previous action to 0 (no action)
        
        self.rso_df = self._prepare_rso_dataframe(
            rso_df=config["rso_df"],
            max_age_days=self.max_rso_age_days,
            rso_pool_size=self.rso_pool_size
        )
        
        self.rso_objects = self._build_rso_objects(self.rso_df)
        self.time_since_last_observed_s = np.full(
            len(self.rso_objects), 
            self.max_time_since_obs_s, 
            dtype=np.float32
        )

        self.current_candidates = []
        
        self.action_space = spaces.Discrete(self.K + 1)

        # The observation space includes the satellite's state and the states of K RSOs.
        self.observation_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(12 + self.K * self.rso_feature_dim,), 
            dtype=np.float32)
        
        self.last_own_propagated = None
        self.last_candidates = None
        self.last_action = None
        self.last_reward = None
        self.last_selected_rso = None

        self._fig = None
        self._ax = None

        self.render_mode = config.get("render_mode", None)

        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"], \
            f"Invalid render_mode: {self.render_mode}"

        self.debug = False
    
    def _prepare_rso_dataframe(self, 
                               rso_df,
                               max_age_days=30, 
                               rso_pool_size=100):
        """
        Clean and reduce the background RSO DataFrame to a manageable size for the environment.
        This includes filtering by age and sampling a subset of RSOs.
        """
        df = rso_df.copy()

        df["EPOCH_DT"] = pd.to_datetime(df["EPOCH"], 
                                        utc=True, 
                                        errors='coerce')
        
        # The propagator works only using TLE1-2 lines
        df = df[df["TLE_LINE1"].notna() & df["TLE_LINE2"].notna()].copy()

        # Exclude controlled RSOs (e.g., Iridium 33) from the background RSO DataFrame
        controlled_id = int(self.orbital_data["cat_id"])
        df = df[df["NORAD_CAT_ID"] != controlled_id].copy()

        # Avoid very old RSOs that may have decayed or are no longer relevant
        current_time = pd.to_datetime(self.orbital_data["epoch"], utc=True)
        min_epoch = current_time - pd.Timedelta(days=max_age_days)
        df = df[df["EPOCH_DT"] >= min_epoch].copy()

        # Normalize missing values in the DataFrame to avoid issues during propagation
        df.fillna({"OBJECT_TYPE": "UNKNOWN"}, inplace=True)

        # Allowed object types for RSOs in the environment
        allowed_types = ["PAYLOAD", "ROCKET BODY", "DEBRIS", "UNKNOWN"]
        df = df[df["OBJECT_TYPE"].isin(allowed_types)].copy()

        # Focus on objects at the same altitude range as the Iridium satellite (500-1200 km)
        controlled_alt_km = float(self.orbital_data["a"]) - R_E  # Semi-major axis minus Earth's radius

        if "ALTITUDE_MEAN_KM_EST" in df.columns:
            df["ALT_DIFF_KM"] = np.abs(df["ALTITUDE_MEAN_KM_EST"] - controlled_alt_km)
            df = df.sort_values(by="ALT_DIFF_KM")
        elif "SEMIMAJOR_AXIS" in df.columns:
            df["ALTITUDE_KM"] = df["SEMIMAJOR_AXIS"] - R_E
            df["ALT_DIFF_KM"] = np.abs(df["ALTITUDE_KM"] - controlled_alt_km)
            df = df.sort_values(by="ALT_DIFF_KM")

        # Sample a subset of RSOs if the pool size is smaller than the available RSOs
        if len(df) > rso_pool_size:
            df = df.sample(n=rso_pool_size, random_state=42).reset_index(drop=True)

        return df.reset_index(drop=True)
    
    def _build_rso_objects(self, rso_df):
        """
        Build a list of RSO objects from the cleaned and 
        reduced DataFrame.

        Each RSO object will contain its orbital data, propagator,
        and other relevant information.
        """
        rso_objects = []
        for rso_idx, row in rso_df.iterrows():
            rso_data = {
                "local_id": rso_idx,
                'cat_id': row['NORAD_CAT_ID'],
                'epoch': row['EPOCH'],
                'object_name': row['OBJECT_NAME'],
                'object_type': row['OBJECT_TYPE'],
                'tle_line1': row['TLE_LINE1'],
                'tle_line2': row['TLE_LINE2'],
                'a': row.get('SEMIMAJOR_AXIS', np.nan),  # Semi-major axis in km
                'e': row.get('ECCENTRICITY', np.nan),  # Eccentricity
                'i': row.get('INCLINATION', np.nan),  # Inclination in radians
                'raan': row.get('RA_OF_ASC_NODE', np.nan),  # Right ascension of ascending node in radians
                'argp': row.get('ARG_OF_PERICENTER', np.nan),  # Argument of perigee in radians
                'M': row.get('MEAN_ANOMALY', np.nan),  # Mean anomaly in radians
                "propagator": SGP4Propagator(row['TLE_LINE1'], row['TLE_LINE2'])
            }
            rso_objects.append(rso_data)
        return rso_objects
    
    def _object_type_code(self, object_type):
        """
        Encode the object type as an integer code.
        PAYLOAD: 0, ROCKET BODY: 1, DEBRIS: 2, UNKNOWN: 3

        TODO: Update to one-hot encoding if needed in the future.
        """
        type_mapping = {
            "PAYLOAD": 0,
            "ROCKET BODY": 1,
            "DEBRIS": 2,
            "UNKNOWN": 3
        }
        return type_mapping.get(object_type, 3)  # Default to UNKNOWN if not found

    def _propagate_rsos(self, datetime_utc):
        """
        Propagate all considered RSOs
        """
        states = []
        num_valid = 0
        num_failed = 0
        first_errors = []

        for obj in self.rso_objects:
            try:
                r_km, v_km_s = obj["propagator"].propagate(datetime_utc)
                
                r_km = np.asarray(r_km, dtype=np.float32)
                v_km_s = np.asarray(v_km_s, dtype=np.float32)
                
                radius_km = np.linalg.norm(r_km)
                velocity_km_s = np.linalg.norm(v_km_s)
                altitude_km = radius_km - R_E

                states.append(
                    {
                        "local_id": obj["local_id"],
                        "cat_id": obj["cat_id"],
                        "r_km": r_km,
                        "v_km_s": v_km_s,
                        "radius_km": radius_km,
                        "velocity_km_s": velocity_km_s,
                        "object_name": obj["object_name"],
                        "object_type": obj["object_type"],
                        "altitude_km": altitude_km,
                        "a": obj["a"],
                        "e": obj["e"],
                        "i": obj["i"],
                        "raan": obj["raan"],
                        "argp": obj["argp"],
                        "M": obj["M"],
                        "valid": True,
                        "error": None
                    }
                )

                num_valid += 1 

            except Exception as e:
                num_failed += 1

                if len(first_errors) < 5:
                    first_errors.append({
                        "cat_id": obj["cat_id"],
                        "object_name": obj["object_name"],
                        "error": str(e)
                    })

                states.append({
                    "local_id": obj["local_id"],
                    "cat_id": obj["cat_id"],
                    "r_km": np.array([np.nan, np.nan, np.nan], dtype=np.float32),
                    "v_km_s": np.array([np.nan, np.nan, np.nan], dtype=np.float32),
                    "radius_km": np.nan,
                    "velocity_km_s": np.nan,
                    "object_name": obj["object_name"],
                    "object_type": obj["object_type"],
                    "altitude_km": np.nan,
                    "a": obj["a"],
                    "e": obj["e"],
                    "i": obj["i"],
                    "raan": obj["raan"],
                    "argp": obj["argp"],
                    "M": obj["M"],
                    "valid": False,
                    "error": str(e)
                })

        self.last_rso_propagation_debug = {
            "num_rso_objects": len(self.rso_objects),
            "num_valid": num_valid,
            "num_failed": num_failed,
            "first_errors": first_errors
        }

        return states
    
    def _select_candidates(self, rso_states, own_propagated):
        """
        Select K candidate RSOs based on proximity and other criteria.

        own_propagated is the tuple returned by _propagate(), 
        which includes the satellite's position and velocity.

        Example:
            0 = no observation
            1 = observe nearest candidate
            2 = observe second-nearest candidate
            ...
            K = observe K-th nearest candidate
        """
        own_r = own_propagated[0]
        own_v = own_propagated[1]

        candidates = []

        for state in rso_states:

            if not state["valid"]:
                continue

            # Calculate relative position and velocity
            rel_pos = state["r_km"] - own_r
            rel_vel = state["v_km_s"] - own_v
            range_km = np.linalg.norm(rel_pos)
            rel_speed_km_s = np.linalg.norm(rel_vel)

            visible = 1 if range_km <= self.max_obs_range_km else 0

            risk_score = np.exp(
                -0.5 * (range_km / self.risk_distance_scale_km) ** 2
            )

            # relative motion terms for risk scoring and normalize
            #  closing_rate_km_s > 0  → RSO is approaching the controlled satellite
            #  closing_rate_km_s < 0  → RSO is moving away
            closing_rate_km_s = -np.dot(rel_pos, rel_vel) / (range_km + 1e-9)

            closing_score = np.clip(
                closing_rate_km_s / self.closing_speed_scale_km_s, 
                0, 
                1
            )

            range_score = np.exp(
                - 0.5 * (range_km / self.risk_distance_scale_km) ** 2
            )
            
            # Makes risk score more sensitive to closing speed, while still considering range
            risk_score = range_score * (0.7 * closing_score + 0.3)
            risk_score = np.clip(risk_score, 0, 1)

            local_id = state["local_id"]
            time_since_last_obs = np.clip(
                self.time_since_last_observed_s[local_id] / self.max_time_since_obs_s, 
                0, 
                1
            )

            object_type_code = self._object_type_code(state["object_type"])

            observability = np.exp(
                - 0.5 * (range_km / self.max_obs_range_km) ** 2
                )

            feature = np.array([
                rel_pos[0] / self.range_norm_km,
                rel_pos[1] / self.range_norm_km,
                rel_pos[2] / self.range_norm_km,
                rel_vel[0] / self.rel_speed_norm_km_s,
                rel_vel[1] / self.rel_speed_norm_km_s,
                rel_vel[2] / self.rel_speed_norm_km_s,
                range_km / self.range_norm_km,
                rel_speed_km_s / self.rel_speed_norm_km_s,
                closing_rate_km_s / self.closing_speed_scale_km_s,
                visible,
                observability,
                risk_score,
                time_since_last_obs,
                object_type_code
            ], dtype=np.float32)
            
            feature = np.clip(feature, -1, 1)

            candidates.append({
                "local_id": local_id,
                "cat_id": state["cat_id"],
                "object_name": state["object_name"],
                "object_type": state["object_type"],
                "state_km":rel_pos,
                "state_km_s": rel_vel,
                "feature": feature,
                "range_km": range_km,
                "rel_speed_km_s": rel_speed_km_s,
                "closing_rate_km_s": closing_rate_km_s,
                "visible": bool(visible),
                "observability": observability,
                "risk_score": risk_score,
                "time_since_last_obs": time_since_last_obs,
                "object_type_code": object_type_code,
                "valid": True
            })
        
        for c in candidates:
            c["priority"] = self._candidate_priority(c)

        # Focus on nearest RSOs based on priority 
        candidates = sorted(
            candidates, 
            key=lambda x: x["priority"], 
            reverse=True)
        candidates = candidates[:self.K]
        #print("Number of candidates selected:", len(candidates))

        # Pad if fewer than K candidates are available
        while len(candidates) < self.K:
            candidates.append({
                "local_id": -1,
                "cat_id": -1,
                "object_name": "NONE",
                "object_type": "NONE",
                "state_km": np.array([0.0, 0.0, 0.0]),
                "state_km_s": np.array([0.0, 0.0, 0.0]),
                "feature": np.zeros(self.rso_feature_dim, dtype=np.float32),
                "range_km": np.inf,
                "rel_speed_km_s": np.inf,
                "closing_rate_km_s": 0.0,
                "visible": False,
                "observability": 0.0,
                "risk_score": 0.0,
                "time_since_last_obs": 1.0,
                "object_type_code": -1,
                "valid": False
            })

        return candidates
    
    def _build_observation_with_candidates(self, own_propagated, candidates):
        """
        Expands upon _build_observation to include the states of K RSOs in the observation vector.

        Build the observation vector including the satellite's state 
        and the states of K RSOs.
        """
        own_obs = self._build_observation(own_propagated)

        rso_features = np.array([c["feature"] for c in candidates], dtype=np.float32)
        rso_features_flat = rso_features.flatten()

        full_obs = np.concatenate([own_obs, rso_features_flat])
        full_obs = np.clip(full_obs, -1, 1)

        return full_obs.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.timestep_index = 0
        self.current_time = pd.to_datetime(self.orbital_data["epoch"], utc=True)

        self.time_since_last_observed_s[:] = self.max_time_since_obs_s
        self.previous_action = 0

        propagated = self._propagate(self.current_time)
        rso_states = self._propagate_rsos(self.current_time)
        candidates = self._select_candidates(rso_states, propagated)

        self.current_candidates = candidates

        self.last_own_propagated = propagated
        self.last_candidates = candidates
        self.last_action = None
        self.last_reward = None
        self.last_selected_rso = None

        obs = self._build_observation_with_candidates(propagated, candidates)

        info = {
            "time_utc": self.current_time,
            "num_rso_objects": len(self.rso_objects),
            "candidate_cat_ids": [c["cat_id"] for c in candidates],
            "candidate_ranges_km": [float(c["range_km"]) for c in candidates],
            "candidate_names": [c["object_name"] for c in candidates],
            "candidate_types": [c["object_type"] for c in candidates],
            "candidate_visible": [bool(c["visible"]) for c in candidates],
            "candidate_risk_scores": [float(c["risk_score"]) for c in candidates],
            "candidate_time_since_last_obs": [float(c["time_since_last_obs"]) for c in candidates],
            "candidate_observability": [float(c["observability"]) for c in candidates],
        }

        return obs.astype(np.float32), info
    
    def step(self, action):
        self.timestep_index += 1
        self.current_time += pd.to_timedelta(self.timestep, unit='s')

        # Every object is propagated at each timestep, and the candidates are selected based on the current state of the satellite and the RSOs.
        self.time_since_last_observed_s += self.timestep  # Increment time since last observed for all RSOs
        self.time_since_last_observed_s = np.clip(
            self.time_since_last_observed_s, 
            0, 
            self.max_time_since_obs_s)

        # Propagate controlled satellite and RSOs, then select candidates for observation
        propagated = self._propagate(self.current_time)
        rso_states = self._propagate_rsos(self.current_time)

        # Select K candidate RSOs based on proximity and other criteria
        candidates = self._select_candidates(rso_states, propagated)
        self.current_candidates = candidates

        num_validate_candidates = sum(c["valid"] for c in candidates)
        finite_ranges = [
            float(c["range_km"]) for c in candidates 
            if np.isfinite(c["range_km"])
        ]

        # Compute reward based on the selected action and the candidates
        reward = self._reward(action, candidates)

        # If the action corresponds to observing a valid and visible RSO, 
        # reset its time since last observed
        #  visible = diagnostic hard flag
        #  observability = reward-relevant soft visibility
        if action > 0:
            selected_rso = candidates[action - 1]
            
            if selected_rso["valid"] and selected_rso["observability"] > 0.1:
                local_id = selected_rso["local_id"]
                self.time_since_last_observed_s[local_id] = 0.0  # Reset time since last observed for the selected RSO

        obs = self._build_observation_with_candidates(propagated, candidates)

        terminated = False
        truncated = self.timestep_index >= self.true_episode_length

        self.last_own_propagated = propagated
        self.last_candidates = candidates
        self.last_action = action
        self.last_reward = reward
        self.last_selected_rso = candidates[action - 1] if action > 0 else None

        info = self._build_info(action, reward, candidates, self.last_selected_rso)

        self.previous_action = action

        if self.debug:
            print("Reward check:")
            for a in range(self.action_space.n):
                r_test = self._reward(a, candidates)

                if a == 0:
                    name = "No Observation"
                    priority = None
                else:
                    c = candidates[a - 1]
                    name = c["object_name"]
                    priority = self._candidate_priority(c)

                if priority is None:
                    print(f" action {a} ({name:20s}): reward = {r_test:.3f}")
                else:
                    print(
                        f" action {a} ({name:20s}): "
                        f"reward = {r_test:.3f}, "
                        f"priority = {priority:.3f}, "
                        f"risk = {c['risk_score']:.3f}, "
                        f"obs = {c['observability']:.3f}, "
                        f"stale = {c['time_since_last_obs']:.3f}"
                    )

        return obs.astype(np.float32), reward, terminated, truncated, info

    def _build_info(self, action, reward, candidates, selected_rso=None):
        """
        Build the info dictionary for the current step.
        """
        selected_cat_id = -1
        selected_object_name = ""
        selected_object_type = ""
        selected_visible = False
        selected_risk_score = 0.0
        selected_time_since_last_obs = 0.0
        selected_observability = 0.0
        selected_priority = 0.0

        if selected_rso is not None:
            selected_cat_id = int(selected_rso.get("cat_id", -1))
            selected_object_name = str(selected_rso.get("object_name", ""))
            selected_object_type = str(selected_rso.get("object_type", ""))
            selected_visible = bool(selected_rso.get("visible", False))
            selected_risk_score = float(selected_rso.get("risk_score", 0.0))
            selected_time_since_last_obs = float(selected_rso.get("time_since_last_obs", 0.0))
            selected_observability = float(selected_rso.get("observability", 0.0))
            selected_priority = float(self._candidate_priority(selected_rso))

        info = {
            "time_utc": self.current_time,
            "num_rso_objects": len(self.rso_objects),
            "candidate_cat_ids": [c["cat_id"] for c in candidates],
            "candidate_ranges_km": [float(c["range_km"]) for c in candidates],
            "candidate_names": [c["object_name"] for c in candidates],
            "candidate_types": [c["object_type"] for c in candidates],
            "candidate_visible": [bool(c["visible"]) for c in candidates],
            "candidate_risk_scores": [float(c["risk_score"]) for c in candidates],
            "candidate_time_since_last_obs": [float(c["time_since_last_obs"]) for c in candidates],
            "selected_action": action,
            "reward": reward,
            "selected_cat_id": selected_cat_id,
            "selected_object_name": selected_object_name,
            "selected_object_type": selected_object_type,
            "selected_visible": selected_visible,
            "selected_risk_score": selected_risk_score,
            "selected_time_since_last_obs": selected_time_since_last_obs,
            "selected_observability": selected_observability,
            "selected_priority": selected_priority,
            "max_priority": max(self._candidate_priority(c) for c in candidates) if candidates else 0.0,
            "priority_regret": max(self._candidate_priority(c) for c in candidates) - selected_priority if selected_rso else 0.0,
        }
        return info

    def _type_priority(self, object_type):
        object_type = str(object_type).upper()

        if object_type == "DEBRIS":
            return 1.0  # Highest priority for debris
        elif object_type == "ROCKET BODY":
            return 0.8  # Medium priority for rocket bodies
        elif object_type == "PAYLOAD":
            return 0.5  # Lower priority for payloads
        elif object_type == "UNKNOWN":
            return 0.6  # Lowest priority for unknown types
        else:
            return 0.4  # Default priority for unrecognized types
    
    def _candidate_priority(self, candidate):
        """
        Instead of an artificial rank reward, we have a priority score based on risk, observability, and staleness.
        
        Priority combines:
        - geometric risk score (based on range and closing speed)
        - observability score (how easy it is to observe the RSO)
        - time since last observation (to encourage observing RSOs that haven't been observed recently)
        - object type priority (to encourage observing certain types of RSOs)
        
        This way, each candidate is rewarded based on its intrinsic properties rather than its rank among the candidates.
        """

        risk_score_term = float(candidate["risk_score"])
        observability_term = float(candidate["observability"])
        time_since_last_obs_term = float(candidate["time_since_last_obs"])
        object_type_priority_term = self._type_priority(candidate["object_type"])

        if not np.isfinite(risk_score_term):
            risk_score_term = 0.0
        if not np.isfinite(observability_term):
            observability_term = 0.0
        if not np.isfinite(time_since_last_obs_term):
            time_since_last_obs_term = 0.0
        if not np.isfinite(object_type_priority_term):
            object_type_priority_term = 0.0

        priority = (
            self.w_risk * risk_score_term +
            self.w_observability * observability_term +
            self.w_staleness * time_since_last_obs_term +
            self.w_type * object_type_priority_term
        )

        if not np.isfinite(priority):
            return 0.0

        return float(priority)

    def _reward(self, action, candidates):
        """
        Reward the agent for selecting useful nearby RSOs

        Action:
            0 = no observation
            1..K = observe candidate[K - 1]
        """
        valid_candidates = [c for c in candidates if c["valid"]]

        if len(valid_candidates) == 0:
            # No valid candidates, reward is zero
            # Penalty for selecting an invalid RSO
            return 0.0 if action == 0 else -1.0  

        priorities = [
            self._candidate_priority(c)
            for c in candidates
            if c["visible"] and c["valid"]
        ]
        max_priority = max(priorities) if priorities else 0.0
        if not np.isfinite(max_priority):
            max_priority = 0.0

        if action == 0:
            # Penalty for not observing when there are visible RSOs
            return (
                -self.noop_priority_penalty * max_priority 
                - self.noop_base_penalty
            )

        selected = candidates[action - 1]

        if not selected["valid"]: # or not selected["visible"]
            return -1.0  # Penalty for selecting an invalid or non-visible RSO
        
        selected_priority = self._candidate_priority(selected)
        if not np.isfinite(selected_priority):
            selected_priority = 0.0

        # Missed priority penalty: if the selected RSO has lower priority
        # than the highest priority visible RSO
        missed_priority_penalty = self.w_missed_priority * max(
            0, 
            max_priority - selected_priority
        )

        # Retasking penalty: if the selected RSO is different from the previous action
        retask_penalty = (
            self.retask_penalty 
            if action != self.previous_action 
            else 0.0
        )

        # Weak penalty for extremely low observability
        low_observability_penalty = 0.0
        selected_observability = float(selected["observability"])
        if not np.isfinite(selected_observability):
            selected_observability = 0.0

        if selected_observability < 0.1:
            low_observability_penalty = 0.1 * (0.1 - selected_observability)

        reward = selected_priority - missed_priority_penalty - retask_penalty - low_observability_penalty

        if not np.isfinite(reward):
            reward = 0.0

        self.previous_action = action
        return float(reward)
    
    def render(self):
        if self.render_mode is None:
            return None
        
        if self.last_candidates is None:
            return None
        
        if self.render_mode == "ansi":
            return self._render_ansi()
        
        if self.render_mode in ["human", "rgb_array"]:
            return self._render_plot()
    
    def _render_ansi(self):
        """
        Render the environment in ANSI mode (text-based).
        """
        lines = []
        lines.append("=" * 40)
        lines.append("Baseline rendering")
        lines.append(f"Time: {self.current_time}")
        lines.append(f"Timestep: {self.timestep_index}")
        lines.append(f"Action taken: {self.last_action}")
        lines.append(f"Reward received: {self.last_reward}")
        lines.append("Candidates:")
        
        for idx, c in enumerate(self.last_candidates):
            lines.append(
                f"  [{idx}] {c['object_name']} | "
                f"Range: {c['range_km']:.2f} km | "
                f"Rel Speed: {c['rel_speed_km_s']:.2f} km/s | "
                f"Risk: {c['risk_score']:.3f} | "
                f"Obs: {c['observability']:.3f} | "
                f"Stale: {c['time_since_last_obs']:.3f} | "
                f"Type: {c['object_type']} | "
                f"Valid: {c['valid']}"
            )
        
        return "\n".join(lines)
    
    def _render_plot(self):
        candidates = self.last_candidates

        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(7, 7))

        ax = self._ax
        ax.clear()

        # Controlled satellite at relative origin.
        ax.scatter(0.0, 0.0, s=120, marker="*", label=f"{self.config['data']['object_name']} (Controlled)", color="blue")

        xs = []
        ys = []
        labels = []
        priorities = []
        sizes = []

        for i, c in enumerate(candidates, start=1):
            if not c["valid"]:
                continue

            dr = np.asarray(c["state_km"], dtype=float)

            xs.append(dr[0])
            ys.append(dr[1])
            labels.append(f"{i}: {c['object_name']}")
            priorities.append(self._candidate_priority(c))

            # Larger marker for higher-priority candidates.
            sizes.append(40.0 + 160.0 * float(self._candidate_priority(c)))

        xs = np.array(xs)
        ys = np.array(ys)

        if len(xs) > 0:
            sc = ax.scatter(xs, ys, s=sizes)

            for x, y, label in zip(xs, ys, labels):
                ax.text(x, y, label, fontsize=8)

        # Draw candidate observation range circle if available.
        if hasattr(self, "max_obs_range_km"):
            circle = plt.Circle(
                (0.0, 0.0),
                self.max_obs_range_km,
                fill=False,
                linestyle="--",
                linewidth=1.0,
            )
            ax.add_patch(circle)

        # Axis limits.
        max_extent = self.max_obs_range_km

        if len(xs) > 0:
            max_candidate_extent = float(
                max(np.max(np.abs(xs)), np.max(np.abs(ys)), self.max_obs_range_km)
            )
            max_extent = 1.1 * max_candidate_extent

        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("Relative x position [km]")
        ax.set_ylabel("Relative y position [km]")

        title = f"SingleSatSSAEnv | step={self.timestep_index}"

        if self.last_action is not None:
            title += f" | action={self.last_action}"

        if self.last_reward is not None:
            title += f" | reward={self.last_reward:.3f}"

        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc="upper right")

        self._fig.tight_layout()

        if self.render_mode == "human":
            plt.pause(0.001)
            return None

        if self.render_mode == "rgb_array":
            self._fig.canvas.draw()

            width, height = self._fig.canvas.get_width_height()
            image = np.frombuffer(self._fig.canvas.tostring_argb(), dtype=np.uint8)
            image = image.reshape((height, width, 4))
            image = image[:, :, [1, 2, 3]]  # Convert ARGB to RGB

            return image
    
    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

if __name__ == "__main__":
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

    mode = input("Enter mode (debug/true/none): ")  # Change to "true" for the actual environment

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
    elif mode == "true":
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
            "render_mode": "rgb_array",}
        env = SingleSatEnv(
            env_config
        )
        check_env(env, skip_render_check=True)

        frames = []

        obs, info = env.reset()
        frames.append(env.render())
        
        assert obs.shape == env.observation_space.shape, "Observation shape mismatch with observation space."
        assert env.action_space.n == env.K + 1, "Action space size mismatch. Expected 6 actions (0-5)."

        for _ in range(10):
            action = env.action_space.sample()  # Sample a random action (for demonstration)
            obs, reward, terminated, truncated, info = env.step(action)
            print("reward:", reward)
            #print("action:", action)
            #print("Observation:", obs)
            print("Selected candidate:", info["selected_object_name"])

            frames.append(env.render())

            if terminated or truncated:
                break
        env.close()

        # Save the frames as a GIF
        gif_filename = "single_sat_env_simulation.gif"
        imageio.mimsave(gif_filename, frames, fps=2)
        print(f"Simulation GIF saved as {gif_filename}")
        
    else:
        print("Goodbye!")