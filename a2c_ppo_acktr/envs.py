import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box
import sys
sys.path.append("..") # Adds higher directory to python modules path.

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
import ICML_envs
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.registration import register as gym_register

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


class Reacher_Fixed(ReacherEnv):
    def __init__(self, mode):
        self.mode = 'Test' if mode else 'Train'
        super().__init__()

    def reset(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        if self.mode == 'Train':
            idx = np.random.choice(3)
            if idx == 0:
                self.goal = np.array([-0.1, -0.1])
            elif idx == 1:
                self.goal = np.array([0.1, -0.1])
            elif idx == 2:
                self.goal = np.array([0.1, 0.1])
        elif self.mode == 'Test':
            self.goal = np.array([-0.1, 0.1])

        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0

        self.set_state(qpos, qvel)
        return self._get_obs()


class Reacher_Fixed_Train(Reacher_Fixed):
    def __init__(self):
        super().__init__(mode=0)


class Reacher_Fixed_Test(Reacher_Fixed):
    def __init__(self):
        super().__init__(mode=1)

gym_register(
    id='ReacherFixedTrain-v0',
    entry_point=Reacher_Fixed_Train,
    max_episode_steps=50,
    reward_threshold=-3.75,
)

gym_register(
    id='ReacherFixedTest-v0',
    entry_point=Reacher_Fixed_Test,
    max_episode_steps=10000,
    reward_threshold=-3.75,
)

class OracleHalfCheetahTest(ICML_envs.OracleHalfCheetah_Friction):

    def __init__(self):
        super().__init__(density_set=[500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150,
                                      1200, 1250, 1300, 1350, 1400, 1450, 1500])

gym_register(
    id="OracleHalfCheetahTrain-v0",
    entry_point=ICML_envs.OracleHalfCheetah_Density,
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

gym_register(
    id="OracleHalfCheetahTest-v0",
    entry_point=OracleHalfCheetahTest,
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

class OracleCartpoleTrain(ICML_envs.OracleRandomCartPole_Force_Length):

    def __init__(self):
        fs = [8, 16, 24, 32, 40]
        ls = [0.1, 0.4, 0.7, 1.10, 1.50, 1.70]
        super().__init__(force_set=fs, length_set=ls)
        # super().__init__(force_set=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        #                  length_set=[0.45, 0.5, 0.55])

gym_register(
    id='OracleCartpoleTrain-v0',
    entry_point=OracleCartpoleTrain,
    max_episode_steps=200,
    reward_threshold=195.0,
)

class OracleCartpoleTrainLMF(ICML_envs.OracleRandomCartPole_Force_Mass_Length):

    def __init__(self):
        super().__init__(force_set=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
                         length_set=[0.45, 0.5, 0.55],
                         mass_set=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

gym_register(
    id='OracleCartpoleTrainLMF-v0',
    entry_point=OracleCartpoleTrainLMF,
    max_episode_steps=200,
    reward_threshold=195.0,
)

class OracleCartpoleTest(ICML_envs.OracleRandomCartPole_Force_Length):

    def __init__(self):
        super().__init__(force_set=[4], length_set=[0.7])

gym_register(
        id='OracleCartpoleTest-v0',
        entry_point=OracleCartpoleTest,
        max_episode_steps=200,
        reward_threshold=195.0,
    )

gym_register(
    id='OracleHumanoid-v0',
    entry_point=ICML_envs.OracleSlimHumanoid,
    max_episode_steps=1000,
)

for mass in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.25, 1.35, 1.45, 1.55]:
    for damp in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.25, 1.35, 1.45, 1.55]:
        gym_register(
            id='OracleSHTest' + str(mass) + "_" + str(damp) + '-v0',
            entry_point=ICML_envs.OracleSlimHumanoid,
            max_episode_steps=1000,
            kwargs={"mass_scale_set": [mass],
                    "damping_scale_set": [damp]}
        )

for mass in [0.30, 0.4, 1.70, 1.80]:
    for damp in [0.30, 0.40, 1.70, 1.80]:
        gym_register(
            id='OracleSHTest' + str(mass) + "_" + str(damp) + '-v0',
            entry_point=ICML_envs.OracleSlimHumanoid,
            max_episode_steps=1000,
            kwargs={"mass_scale_set": [mass],
                    "damping_scale_set": [damp]}
        )

for f in [80, 160, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
    for l in [3.4, 6.8, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.8, 1.85, 1.9]:

        gym_register(
            id='OracleCartpoleTest' + str(f) + "_" + str(l) + '-v0',
            entry_point=ICML_envs.OracleRandomCartPole_Force_Length,
            max_episode_steps=200,
            reward_threshold=195.0,
            kwargs={"force_set": [f], "length_set": [l]}
        )
        # print('OracleCartpoleTest' + str(f) + "_" + str(l) + '-v0')
        for m in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00]:
            gym_register(
                id='OracleCartpoleTestLMF' + str(f) + "_" + str(l) + '_' + str(m) + '-v0',
                entry_point=ICML_envs.OracleRandomCartPole_Force_Mass_Length,
                max_episode_steps=200,
                reward_threshold=195.0,
                kwargs={"force_set": [f], "length_set": [l], 'mass_set': [m]}
            )

for d in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]:
    gym_register(
        id='OracleHalfCheetahTest_' + str(d) + '-v0',
        entry_point=ICML_envs.OracleHalfCheetah_Density,
        max_episode_steps=1000,
        reward_threshold=4800,
        kwargs={'density_set':[d]}
    )

def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():

        if env_id == "SHTrain":
            env = gym.make("OracleHumanoid-v0")
        elif env_id.startswith("HCTrain"):

            env = gym.make("OracleHalfCheetahTrain-v0")
        elif env_id.startswith("HCTest"):

            env = gym.make("OracleHalfCheetahTest-v0")
        elif env_id.startswith("OracleTrain"):

            env = gym.make("OracleCartpoleTrain-v0")

        elif env_id.startswith("OracleTrainLMF"):

            env = gym.make("OracleCartpoleTrainLMF-v0")

        elif env_id == "OracleTest":

            env = gym.make("OracleCartpoleTest-v0")

        elif env_id.startswith("OracleTest"):

            env = gym.make(env_id)

        elif env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
