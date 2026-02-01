from utils import *

class Trainer:
    def __init__(self, env, eval_env, agent, args):
        self.args = args
        self.agent = agent

        self.delayed_env      = env
        self.eval_delayed_env = eval_env

        self.start_step = args.start_step
        self.update_after = args.update_after
        self.max_step = args.max_step
        self.batch_size = args.batch_size
        self.update_every = args.update_every

        self.eval_flag = args.eval_flag
        self.eval_episode = args.eval_episode
        self.eval_freq = args.eval_freq

        self.episode = 0
        self.total_step  = 0
        self.finish_flag = False

        self.local_step = 0
        self.eval_local_step = 0

        # tags for checking the order of state generation
        self.state_idx  = 0
        self.eval_state_idx  = 0

        self.init_obs_delayed_steps  = args.init_obs_delayed_steps
        self.min_obs_delayed_steps   = args.min_obs_delayed_steps
        self.max_obs_delayed_steps   = args.max_obs_delayed_steps

    # training
    def train(self):
        while not self.finish_flag:
            self.episode   += 1
            self.state_idx  = 1
            self.local_step = 0
            self.delayed_env.reset()
            self.agent.temporary_buffer.clear()
            obs_temp, reward_temp, done_temp = [], [], []
            done = False

            # Episode starts here.
            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.local_step < self.init_obs_delayed_steps:  # if t < o_max
                    action = np.zeros_like(self.delayed_env.action_space.sample())  # Select the 'no-ops'
                    obs_list, reward_list, done_list, _ = self.delayed_env.step(action)
                    self.agent.temporary_buffer.actions.append(action) # Put a(t) to the temporary buffer B

                    if len(obs_list) > 0: # append observed states, rewards, dones
                        for obs, rwd, dn in zip(obs_list, reward_list, done_list):
                            obs_temp.append(obs)
                            reward_temp.append(rwd)
                            done_temp.append(dn)

                elif self.local_step == self.init_obs_delayed_steps:  # if t == o_max
                    action = np.zeros_like(self.delayed_env.action_space.sample())  # Select the 'no-ops'
                    obs_list, reward_list, done_list, _ = self.delayed_env.step(action)

                    if len(obs_list) > 0:  # append observed states, rewards, dones
                        for obs, rwd, dn in zip(obs_list, reward_list, done_list):
                            obs_temp.append(obs)
                            reward_temp.append(rwd)
                            done_temp.append(dn)

                    # sort observed states in generated order
                    obs_temp, reward_temp, done_temp = do_sorting(obs_temp,reward_temp,done_temp)

                    self.agent.temporary_buffer.actions.append(action)  # Put a(t) to the temporary buffer B
                    self.agent.temporary_buffer.states.append(obs_temp[0][0]) # Put most recent usable state to the temporary buffer B
                    self.state_idx += 1

                    # delete used components
                    obs_temp, reward_temp, done_temp  = eliminate_head(obs_temp,reward_temp,done_temp)

                else:  # if t > o_max
                    # construct augmented state x(t) and select action a(t)
                    most_recent_usable_state = self.agent.temporary_buffer.states[-1]
                    first_action_idx = len(self.agent.temporary_buffer.actions) - self.init_obs_delayed_steps
                    augmented_state = self.agent.temporary_buffer.get_augmented_state(most_recent_usable_state, first_action_idx)
                    action = self.agent.get_action(augmented_state, evaluation=False)

                    obs_list, reward_list, done_list, _ = self.delayed_env.step(action)

                    if len(obs_list) > 0:  # append observed states, rewards, dones
                        for obs, rwd, dn in zip(obs_list, reward_list, done_list):
                            obs_temp.append(obs)
                            reward_temp.append(rwd)
                            done_temp.append(dn)

                    # sort observed states in order of generation
                    obs_temp, reward_temp, done_temp = do_sorting(obs_temp, reward_temp, done_temp)

                    # for the most recent usable state
                    if len(obs_temp) > 0 and self.state_idx == obs_temp[0][1]:
                        next_state = obs_temp[0][0]
                        reward     = reward_temp[0][0]
                        done       = done_temp[0][0]
                        self.agent.temporary_buffer.actions.append(action)    # Put a(t) to the temporary buffer B
                        self.agent.temporary_buffer.states.append(next_state) # Put most recent usable state to the temporary buffer B
                        self.state_idx += 1

                        # Delete used ones
                        obs_temp, reward_temp, done_temp = eliminate_head(obs_temp, reward_temp, done_temp)

                        true_done = 0.0 if self.local_step == self.delayed_env._max_episode_steps + self.init_obs_delayed_steps else float(done)
                        if len(self.agent.temporary_buffer.states) >= (self.init_obs_delayed_steps + 2):
                            augmented_s, s, a, next_augmented_s, next_s = self.agent.temporary_buffer.get_tuple()
                            self.agent.replay_memory.push(augmented_s, s, a, reward, next_augmented_s, next_s, true_done)
                    else:
                        raise ValueError("Check init_obs_delay == max_obs_delay.")

                # Update parameters
                if self.agent.replay_memory.size >= self.batch_size \
                        and self.total_step >= self.update_after and \
                        self.total_step % self.update_every == 0:

                    for i in range(self.update_every):
                        self.agent.train()

                # Evaluate
                if self.eval_flag and self.total_step % self.eval_freq == 0:
                    self.evaluate()

                # Finish flag
                if self.total_step == self.max_step:
                    self.finish_flag = True

    # evaluation
    def evaluate(self):
        reward_list = []
        for epi in range(self.eval_episode):
            self.eval_local_step = 0
            self.eval_state_idx  = 1
            self.eval_delayed_env.reset()
            self.agent.eval_temporary_buffer.clear()
            eval_obs_temp, eval_reward_temp, eval_done_temp = [], [], []
            episode_reward = 0
            done = False

            while not done:
                self.eval_local_step += 1
                if self.eval_local_step < self.init_obs_delayed_steps:
                    action = np.zeros_like(self.eval_delayed_env.action_space.sample())
                    eval_obs_list, eval_reward_list, eval_done_list, _ = self.eval_delayed_env.step(action)
                    self.agent.eval_temporary_buffer.actions.append(action)
                    if len(eval_obs_list) > 0:
                        for obs, rwd, dn in zip(eval_obs_list, eval_reward_list, eval_done_list):
                            eval_obs_temp.append(obs)
                            eval_reward_temp.append(rwd)
                            eval_done_temp.append(dn)
                elif self.eval_local_step == self.init_obs_delayed_steps:
                    action = np.zeros_like(self.eval_delayed_env.action_space.sample())
                    eval_obs_list, eval_reward_list, eval_done_list, _ = self.eval_delayed_env.step(action)
                    if len(eval_obs_list) > 0:
                        for obs, rwd, dn in zip(eval_obs_list, eval_reward_list, eval_done_list):
                            eval_obs_temp.append(obs)
                            eval_reward_temp.append(rwd)
                            eval_done_temp.append(dn)
                    eval_obs_temp, eval_reward_temp, eval_done_temp = do_sorting(eval_obs_temp,
                                                                                 eval_reward_temp,
                                                                                 eval_done_temp)
                    episode_reward += eval_reward_temp[0][0]
                    self.agent.eval_temporary_buffer.actions.append(action)
                    self.agent.eval_temporary_buffer.states.append(eval_obs_temp[0][0])
                    self.eval_state_idx += 1

                    # Delete used ones
                    eval_obs_temp, eval_reward_temp, eval_done_temp = eliminate_head(eval_obs_temp,
                                                                                     eval_reward_temp,
                                                                                     eval_done_temp)

                else:
                    most_recent_usable_state = self.agent.eval_temporary_buffer.states[-1]
                    first_action_idx    = len(self.agent.eval_temporary_buffer.actions) - self.init_obs_delayed_steps
                    augmented_state     = self.agent.eval_temporary_buffer.get_augmented_state(most_recent_usable_state, first_action_idx)
                    action = self.agent.get_action(augmented_state, evaluation=True)

                    eval_obs_list, eval_reward_list, eval_done_list, _ = self.eval_delayed_env.step(action)
                    if len(eval_obs_list) > 0:
                        for obs, rwd, dn in zip(eval_obs_list, eval_reward_list, eval_done_list):
                            eval_obs_temp.append(obs)
                            eval_reward_temp.append(rwd)
                            eval_done_temp.append(dn)
                    eval_obs_temp, eval_reward_temp, eval_done_temp = do_sorting(eval_obs_temp,
                                                                                 eval_reward_temp,
                                                                                 eval_done_temp)

                    if len(eval_obs_temp) > 0 and self.eval_state_idx == eval_obs_temp[0][1]:
                        episode_reward += eval_reward_temp[0][0]
                        done = eval_done_temp[0][0]
                        self.agent.eval_temporary_buffer.actions.append(action)
                        self.agent.eval_temporary_buffer.states.append(eval_obs_temp[0][0])
                        self.eval_state_idx += 1

                        # Delete used ones
                        eval_obs_temp, eval_reward_temp, eval_done_temp = eliminate_head(eval_obs_temp,
                                                                                         eval_reward_temp,
                                                                                         eval_done_temp)
                    else:
                        raise ValueError("Check init_obs_delay == max_obs_delay.")

            reward_list.append(episode_reward)
        log_to_txt(self.args.delay_type, self.args.env_name, self.args.random_seed,  self.max_obs_delayed_steps, self.total_step, sum(reward_list) / len(reward_list))
        print("Eval  |  Total Steps {}  |  Episodes {}  |  Average Reward {:.2f}  |  Max reward {:.2f}  |  "
              "Min reward {:.2f}".format(self.total_step, self.episode, sum(reward_list) / len(reward_list), max(reward_list), min(reward_list)))






