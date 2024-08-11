import gymnasium
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
import numpy as np

def convert_numpy_to_python(data):
    if isinstance(data, dict):
        return {key: convert_numpy_to_python(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_python(element) for element in data]
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    else:
        return data

class RobloxBedwars(gymnasium.Env):
    """Custom Environment that follows the OpenAI Gym interface"""

    def __init__(self):
        super(RobloxBedwars, self).__init__()
        # dict isn't supported for action spaces
        # Define action space
        # 0: jump (0 or 1)
        # 1: hit (0 or 1)
        # 2-5: wasd
        self.action_space = MultiDiscrete([2, 2, 2, 2, 2, 2])
        self.observation_space = Dict({
            'my_position': Box(low=-1, high=1, shape=(3,)),
            'my_rotation': Box(low=0, high=360, shape=(3,)),
            'my_health': Discrete(101),
            'their_position': Box(low=-1000, high=1000, shape=(3,)),
            'their_rotation': Box(low=0, high=360, shape=(3,)),
            'their_health': Discrete(101),
            'raycast_distances': Box(low=0, high=100, shape=(74,)),
            'hit_types': Box(low=0, high=3, shape=(74,)),
            'hit_positions': Box(low=-1000, high=1000, shape=(222,)),
        })
        self.reset_character = False
        self.calculate_reward_poll = None
        self.take_action_poll = None
        self.get_observation  = None
        self.steps_beyond_done = None

    def reset(self, seed=None, options=None):
        self.steps_beyond_done = None
        self.reset_character = True
        self.get_observation = None
        while self.reset_character and not self.get_observation:
            continue
        observation = self.get_observation
        info = {}
        return observation, info

    def step(self, action):
        self.calculate_reward_poll = None
        self.take_action_poll = convert_numpy_to_python({
            'jump': action[0] == 1, # 1 is true, 0 is false
            'hit': action[1] == 1,
            'w': action[2] == 1,
            'a': action[3] == 1,
            's': action[4] == 1,
            'd': action[5] == 1
        })
        self.get_observation = None
        while self.calculate_reward_poll is None or self.take_action_poll is not None or not self.get_observation:
            continue
        done = self.get_observation['my_health'] == 0 or self.get_observation['their_health'] == 0
        truncated = False
        reward = self.calculate_reward_poll
        observation = self.get_observation
        return observation, reward, done, truncated, {}

    def poll_callback(self, data):
        if data['calculate_reward_poll'] is not False:
            self.calculate_reward_poll = data['calculate_reward_poll']
        if data['reset_character']:
            self.reset_character = False
        if data['take_action_poll']:
            self.take_action_poll = None
        if data['get_observation']:
            self.get_observation = data['get_observation']
    