import gym
import ray
from ray import tune
import os
import gridworld_gym
from gridworld_gym.envs import GridWorldEnv as env_creator

# configuration and init
log_dir = "logs/"
log_path = os.path.join(os.getcwd(), log_dir)
seed = 123
max_iter = 500  # how many iterations to train since no stop criterion is set
##########################################################################################
# Ray inits
ray.init(local_mode=True, ignore_reinit_error=True)  # local mode since it is not run on a gpu
RAY_IGNORE_UNHANDLED_ERRORS = 1

print("--Start RL--")
# environment needs to be registered since it is a custom gym with no predefined constructor
tune.register_env("gridworld-v0", env_creator)
##########################################################################################

# show grid - rendering not used during tuning since it decreases performance
env_ex = gym.make("gridworld-v0")
print(env_ex.render(mode='human', close=False))
del env_ex

# configuration sampled from https://docs.ray.io/en/latest/rllib/rllib-training.html and changed accordingly
tune.run("PPO",
         config={
             "env": "gridworld-v0",
             "num_gpus": 0,  # 1 => with gpu
             "seed": seed,
             "horizon": 400,  # end gym iteration after 400 steps instead of 800
             "soft_horizon": False,
             "render_env": False,  # change if env should be rendered
         },
         local_dir=log_dir,
         verbose=3,
         stop=ray.tune.stopper.MaximumIterationStopper(max_iter),
         reuse_actors=True
         )

pass
