from abc import ABC, abstractmethod
import torch
import numpy

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, aux_info):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        aux_info : list
            a list of strings corresponding to the name of the extra information
            retrieved from the environment for supervised auxiliary losses

        """
        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.acmodel.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.aux_info = aux_info

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc // 2 * self.num_procs

        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc // 2, self.num_procs)

        self.obs, self.human_view = self.env.reset()  # fk: resets all parallel environments

        # fk: self.obs_old is needed for the sampled obs from the buffer because self.obs is needed for
        # collect_new_experiences(). self.obs_old is initialized with self.obs, however, these initial values are not
        # used but overwritten in collect_old_experiences().
        self.obs_old = self.obs
        self.obss_old = [None] * (shape[0])

        self.mission = numpy.array([e['mission'] for e in self.obs])
        self.obss = [None] * (shape[0])
        # 20221229 fk: this holds the image of the first frame of an episode
        self.human_views = numpy.empty(shape, dtype=object)
        # 20230108 fk: this holds the instructions of an episode
        self.missions = numpy.empty(shape, dtype=object)

        self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
        self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)

        self.mask = torch.ones(shape[1], device=self.device)
        self.mask_old = torch.ones(shape[1], device=self.device)  # fk: needed for collect_old_experiences()
        self.masks = torch.zeros(*shape, device=self.device)
        self.masks_old = torch.zeros(*shape, device=self.device)  # fk: needed for collect_old_experiences()
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # fk: data structure to hold original rewards (needed for feedback integration):
        self.orig_rewards = torch.zeros(*shape, device=self.device)

        # 20221224 fk: added this to pass agent position and direction and grid size to frontend:
        self.grid_width = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.grid_height = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.agent_x_pos = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.agent_y_pos = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.agent_dir = torch.zeros(*shape, device=self.device, dtype=torch.int)

        if self.aux_info:
            self.aux_info_collector = ExtraInfoCollector(self.aux_info, shape, self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_new_experiences(self):
        """This is an adaptation of the original collect_experiences method from BabyAI Code. Here, only
        self.num_frames_per_proc // 2 frames per process are collected since for PPO the other
        self.num_frames_per_proc frames per process are sampled from the already collected experiences in the buffer
        (see collect_old_experiences method below).

        Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc. as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc // 2` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """

        # fk: list of lists of end indexes of episodes. Each list contains the end indexes of the episodes of one of the
        # self.num_proc processes. We use this list to track when episodes are done.
        episodes_end_indexes = [[] for _ in range(self.num_procs)]

        for i in range(self.num_frames_per_proc // 2):
            # Do one agent-environment interactions

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']

            action = dist.sample()

            obs, reward, done, env_info = self.env.step(action.cpu().numpy())

            # 20221224 fk: hand over gridsize and agent position and direction to frontend
            env_info = list(zip(*list(env_info)))

            self.grid_width[i] = torch.tensor(env_info[2], device=self.device)
            self.grid_height[i] = torch.tensor(env_info[3], device=self.device)
            self.agent_x_pos[i] = torch.tensor(numpy.array(env_info[0]).T[:1], device=self.device)
            self.agent_y_pos[i] = torch.tensor(numpy.array(env_info[0]).T[1:2], device=self.device)
            self.agent_dir[i] = torch.tensor(env_info[1], device=self.device)

            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)

            # Update experiences values
            self.missions[i] = self.mission
            self.human_views[i] = self.human_view
            self.obss[i] = self.obs
            self.obs = obs
            self.mission = numpy.array([e['mission'] for e in self.obs])
            self.human_view = env_info[4]

            self.memories[i] = self.memory
            self.memory = memory

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            self.orig_rewards[i] = torch.tensor(reward, device=self.device)
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            if self.aux_info:
                self.aux_info_collector.fill_dictionaries(i, env_info, extra_predictions)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for j, done_ in enumerate(done):
                if done_:
                    # fk: when an episode is done, add end index of the episode to episodes_end_indexes:
                    episodes_end_indexes[j].append(i)

                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[j].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[j].item())
                    self.log_num_frames.append(self.log_episode_num_frames[j].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            next_value = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))['value']

        for i in reversed(range(self.num_frames_per_proc // 2)):
            next_mask = self.masks[i + 1] if i < self.num_frames_per_proc // 2 - 1 else self.mask
            next_value = self.values[i + 1] if i < self.num_frames_per_proc // 2 - 1 else next_value
            next_advantage = self.advantages[i + 1] if i < self.num_frames_per_proc // 2 - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc // 2)]

        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # fk: add original rewards to exps:
        exps.orig_reward = self.orig_rewards.transpose(0, 1).reshape(-1)

        # 20221227 fk: add agent position, agent direction and grid size to exps:
        exps.agent_x_pos = self.agent_x_pos.transpose(0, 1).reshape(-1)
        exps.agent_y_pos = self.agent_y_pos.transpose(0, 1).reshape(-1)
        exps.agent_dir = self.agent_dir.transpose(0, 1).reshape(-1)
        exps.grid_width = self.grid_width.transpose(0, 1).reshape(-1)
        exps.grid_height = self.grid_height.transpose(0, 1).reshape(-1)

        # 20221229 fk: flatten imgs and missions:
        exps.human_views = self.human_views.transpose().reshape(-1)  # TODO: adjust to include images
        exps.missions = self.missions.transpose().reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log, episodes_end_indexes

    def collect_old_experiences(self, sampled_actions, sampled_values, sampled_masks, sampled_memories,
                                sampled_rewards, sampled_orig_rewards, sampled_obs, sampled_returnn):
        """This is an adaptation of the original collect_experiences method from BabyAI Code. Note that only
        self.num_frames_per_proc // 2 frames per process are used since for PPO the other
        self.num_frames_per_proc frames per process are newly collected experiences (see collect_new_experiences method
        above).

        Computes new advantages for episodes sampled from the buffer.

        The rollouts and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc. as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc // 2` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!

        """

        for i in range(self.num_frames_per_proc // 2):
            # Do one agent-environment interaction

            # print(f'sampled_rewards{i}: {sampled_rewards[i]}')

            # self.obs_old = sampled_obs[i]
            self.mask_old = sampled_masks[i]

            preprocessed_obs = self.preprocess_obss(sampled_obs[i], device=self.device)
            with torch.no_grad():
                model_results = self.acmodel(preprocessed_obs, self.memory * self.mask_old.unsqueeze(1))
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']

            action = sampled_actions[i]
            reward = sampled_rewards[i]
            env_info = tuple()

            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)

            # Update experiences values
            self.obss_old[i] = self.obs_old
            if i < self.num_frames_per_proc // 2 - 1:
                self.obs_old = sampled_obs[i + 1]  # will be used for next iteration or next_value respectively

            self.memories[i] = self.memory
            self.memory = memory

            self.masks_old[i] = self.mask_old
            if i < self.num_frames_per_proc // 2 - 1:
                self.mask_old = sampled_masks[i + 1]  # will be used for next iteration or next_value respectively

            self.actions[i] = action
            self.values[i] = value
            self.rewards[i] = reward

            if self.aux_info:
                self.aux_info_collector.fill_dictionaries(i, env_info, extra_predictions)

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs_old, device=self.device)
        with torch.no_grad():
            next_value = self.acmodel(preprocessed_obs, self.memory * self.mask_old.unsqueeze(1))['value']

        for i in reversed(range(self.num_frames_per_proc // 2)):
            next_mask = self.masks_old[i + 1] if i < self.num_frames_per_proc // 2 - 1 else self.mask_old
            next_value = self.values[i + 1] if i < self.num_frames_per_proc // 2 - 1 else next_value
            next_advantage = self.advantages[i + 1] if i < self.num_frames_per_proc // 2 - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()
        exps.obs = [self.obss_old[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc // 2)]

        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks_old.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        # exps.returnn = exps.value + exps.advantage
        exps.returnn = sampled_returnn.transpose(0, 1).reshape(-1)
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # fk: add original rewards to exps:
        # remove last frame from sampled_orig_rewards (because it is not used for PPO):
        sampled_orig_rewards = sampled_orig_rewards[:, :-1]
        exps.orig_reward = sampled_orig_rewards.transpose(0, 1).reshape(-1)

        # fk: add empty agent position, agent direction and grid size to exps (because in ppo.py the newly collected
        # exps and the old exps are concatenated for PPO and the newly collected exps still need to be added to the
        # buffer and therefore need to contain agent position, agent direction and grid size):
        exps.agent_x_pos = torch.zeros(self.num_frames_per_proc // 2 * self.num_procs, device=self.device)
        exps.agent_y_pos = torch.zeros(self.num_frames_per_proc // 2 * self.num_procs, device=self.device)
        exps.agent_dir = torch.zeros(self.num_frames_per_proc // 2 * self.num_procs, device=self.device)
        exps.grid_width = torch.zeros(self.num_frames_per_proc // 2 * self.num_procs, device=self.device)
        exps.grid_height = torch.zeros(self.num_frames_per_proc // 2 * self.num_procs, device=self.device)

        # fk: flatten empty imgs and missions and add them to exps (for same reason as with agent position etc. above):
        exps.human_views = numpy.empty(self.num_frames_per_proc // 2 * self.num_procs, dtype=object)
        exps.missions = numpy.empty(self.num_frames_per_proc // 2 * self.num_procs, dtype=object)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        return exps

    @abstractmethod
    def update_parameters(self, q):  # fk: added q
        pass
