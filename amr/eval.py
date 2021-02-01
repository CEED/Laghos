
import gym
from gym import spaces
import ray
import ray.rllib.agents.ppo as ppo

import numpy as np

# Note that rllib does require you to give the policy an env with the
# same action and observation spaces as used in training. The rest of
# it can be "fake" if you provide your own observation data some other
# way.

class DummyEnv(gym.Env):
    def __init__(self, config): # the config param is required by rllib
        self.num_prizes = 8

        # choose one by index
        self.action_space = spaces.Discrete(self.num_prizes)

        # range is [0,1]
        self.observation_space = spaces.Box(0.0, 1.0,
                                            shape=(self.num_prizes,),
                                            dtype=np.float32)
        self.state = None
        
    def step(self, action):
        pass
    def reset(self):
        pass
    def render(self):
        pass

class Evaluator():

    def __init__(self):

        ray.shutdown()
        ray.init()

        config = ppo.DEFAULT_CONFIG.copy()
        config["log_level"] = "WARN"

        # Create agent from checkpoint
        self.agent = ppo.PPOTrainer(config,env=DummyEnv)
        #self.agent.restore("/g/g10/rwa/ray_results/PPO_PickPrizeGame_2021-01-30_21-35-06v763ve7j/checkpoint_1/checkpoint-1")

        self.env = DummyEnv({})

    def eval(self,obs):
        print("evaluating policy")
        print("obs: ",obs)
        pick = self.agent.compute_action(obs)
        print("action: ",pick)
        return pick
