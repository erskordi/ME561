import numpy as np

def no_op_policy(obs, info):
    """
    A simple policy that always chooses the no-op action (0).
    """
    return 0

def random_policy(obs, info, env):
    """
    A simple policy that randomly selects an action from the available actions.
    """
    return env.action_space.sample()  # Randomly sample an action from the action space

def max_priority_policy(candidates, env):
    """
    A policy that selects the RSO with the highest priority from the visible candidates.
    If no candidates are visible, it chooses the no-op action (0).
    """
    priorities = [env._candidate_priority(c) for c in candidates]
    return int(np.argmax(priorities)) + 1 if candidates else 0  # +1 because action 0 is no-op