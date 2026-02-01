import random
import gym
from utils import *

class DelayedEnv(gym.Wrapper):
    def __init__(self, args, env, seed):

        super(DelayedEnv, self).__init__(env)
        assert args.max_obs_delayed_steps > 0,  'args.max_obs_delayed_steps must be greater than 0'
        assert args.init_obs_delayed_steps > 0, 'args.init_obs_delayed_steps must be greater than 0'
        self.env.action_space.seed(seed)
        self._max_episode_steps = self.env._max_episode_steps

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.delay_type = args.delay_type
        self.mu = args.mu

        self.obs_buffer    = []
        self.reward_buffer = []
        self.done_buffer   = []

        self.init_obs_delayed_steps = args.init_obs_delayed_steps
        self.min_obs_delayed_steps  = args.min_obs_delayed_steps
        self.max_obs_delayed_steps  = args.max_obs_delayed_steps

        self.state_idx  = None
        self.init_delay = None
        self.init_state = None

    def reset(self, seed=None, option=None):
        init_state, _ = self.env.reset()
        self.init_state = init_state
        self.state_idx  = 1 # state generation time
        self.local_time = 1 # time-step

        self.init_delay = self.init_obs_delayed_steps

        self.obs_buffer    = []
        self.reward_buffer = []
        self.done_buffer   = []

        # meta obs: (a, b, c) -> a = obs, b = generated time, c = delay
        self.obs_buffer.append([init_state, self.state_idx, self.init_delay])
        self.reward_buffer.append([0, self.state_idx, self.init_delay])
        self.done_buffer.append([False, self.state_idx, self.init_delay])
        return init_state

    def step(self, action):
        current_obs, current_reward, current_terminated, current_truncated, _ = self.env.step(action)
        current_done = current_terminated or current_truncated
        data_dict = {'current_obs': current_obs, 'current_reward': current_reward, 'current_done': current_done}

        self.local_time += 1
        self.state_idx += 1

        if self.delay_type == 'uniform':
            delay = random.randrange(self.min_obs_delayed_steps, self.max_obs_delayed_steps + 1)
        elif self.delay_type == 'Poisson':
            if self.mu == None:
                raise ValueError('Set Poisson parameter')
            delay = poisson_delay(min_delay=self.min_obs_delayed_steps, max_delay=self.max_obs_delayed_steps, mu=self.mu)
        else:
            raise Exception

        # meta obs: (a, b, c) -> a= obs, b=generated time, c= how delayed
        meta_obs    = [current_obs,    self.state_idx, delay]
        meta_reward = [current_reward, self.state_idx, delay]
        meta_done   = [current_done,   self.state_idx, delay]

        # Push the current obs, rwd, done with metadata.
        self.obs_buffer.append(meta_obs)
        self.reward_buffer.append(meta_reward)
        self.done_buffer.append(meta_done)

        ret_meta_obs_list    = []
        ret_meta_reward_list = []
        ret_meta_done_list   = []

        for i, meta_obs in enumerate(self.obs_buffer):
            if meta_obs[1] + meta_obs[2] <= self.local_time:  # check observable states
                ret_meta_obs_list.append(self.obs_buffer[i])
                ret_meta_reward_list.append(self.reward_buffer[i])
                ret_meta_done_list.append(self.done_buffer[i])

        # Pop the observed delayed obs, rwd, done
        for meta_obs in ret_meta_obs_list:
            for i, obs in enumerate(self.obs_buffer):
                if obs[1] == meta_obs[1]:
                    idx = i
            del self.obs_buffer[idx]

        for meta_reward in ret_meta_reward_list:
            for i, rwd in enumerate(self.reward_buffer):
                if rwd[1] == meta_reward[1]:
                    idx = i
            del self.reward_buffer[idx]

        for meta_done in ret_meta_done_list:
            for i, dn in enumerate(self.done_buffer):
                if dn[1] == meta_done[1]:
                    idx = i
            del self.done_buffer[idx]

        return ret_meta_obs_list, ret_meta_reward_list, ret_meta_done_list, data_dict

