import random
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import Wrapper, spaces
from typing import Any

register(id = "EnvironmentAgentTrading", entry_point = "src.environments.wrapper:EnvironmentAgentTradingWrapper")

class EnvironmentAgentTradingWrapper(Wrapper):
    def __init__(self,
                 env: Any,
                 transition_shape = None,
                 seed=42):
        super().__init__(env)
        self.seed = seed

        self.env = env

        random.seed(seed)
        np.random.seed(seed)

        self.action_labels = env.action_labels

        state_shape = transition_shape["states"]["shape"][1:]
        state_type = transition_shape["states"]["type"]

        self.action_space = spaces.Discrete(
            n = env.action_dim,
        )

        self.observation_space = spaces.Dict({
            'states': spaces.Box(low=-np.inf, high=np.inf, shape=state_shape, dtype=state_type),
        })

        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self):
        state, info = self.env.reset()
        return state, info

    def step(self, action):
        next_state, reward, done, truncted, info = self.env.step(action)
        return next_state, reward, done, truncted, info

def make_env(env_id, env_params):
    def thunk():
        env = gym.make(env_id, **env_params)
        return env
    return thunk