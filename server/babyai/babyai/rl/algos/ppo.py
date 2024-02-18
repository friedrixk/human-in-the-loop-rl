import datetime
import random
import time

import numpy
import torch

# fk: imports
import pandas as pd
import pickle

from babyai.rl.algos.base import BaseAlgo
from babyai.rl.utils import DictList


class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, aux_info=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         aux_info)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, (beta1, beta2), eps=adam_eps)
        self.batch_num = 0

        # fk: boolean to track if buffer is usable, i.e. if populate_buffer() has been called twice (before that we
        # cannot sample self.num_frames_per_proc // 2 + 1 frames):
        self.buffer_usable = False

        # fk: track how many episodes are done:
        self.episodes_done = 0

        # fk: List that holds tuples of pandas dataframes and a boolean. The dataframes correspond to the processes and
        # hold the episodes data of each process. The boolean indicates whether the last row of a dataframe is a
        # finished episode (True = episode is finished, False = episode is not finished). This is needed for deciding if
        # newly collected exps have to be appended to the last episode or if a new episode can be appended to the
        # process dataframe.
        self.proc_dfs = [[pd.DataFrame(
            columns=['obs_image', 'obs_instr', 'memories', 'masks', 'actions', 'values', 'rewards', 'orig_rewards',
                     'advantages', 'returnn', 'agent_x_pos', 'agent_y_pos', 'agent_dir', 'grid_width', 'grid_height',
                     'human_views', 'missions', 'entropy_stepwise', 'policy_loss_stepwise', 'value_loss_stepwise',
                     'loss_stepwise', 'feedback']),
            True] for _ in range(self.num_procs)]

        # fk: initialise buffer with buffer_initialisation.pickle:
        with open('buffer_initialisation_00.pkl', 'rb') as f:
            self.proc_dfs = pickle.load(f)
        print('buffer initialised')
        print(f'buffer length: {len(self.proc_dfs[0][0])}')

        # set all booleans in self.proc_dfs to True (to ensure that new exps are appended to a new buffer line):
        # for proc_df in self.proc_dfs:
        #     proc_df[1] = True

        # count how many episodes in the buffer are done:
        for proc_df in self.proc_dfs:
            self.episodes_done += proc_df[0].shape[0]
            # if proc_df[1] == 0 subtract 1 from episodes_done (because the last episode is not done yet):
            if not proc_df[1]:
                self.episodes_done -= 1

        print('episodes done: ', self.episodes_done)

        # fk: track how many frames are stored in the buffer:
        self.frames_in_buffer = 0

        # fk: track how often frames with feedback have been re-integrated into the optimization:
        self.amount_feedback_frames = 0

        # list that holds all process and episode indexes of episodes with feedback:
        self.feedback_episodes = []

        # counter that continuously increases and is used to update the buffer file when it reaches a certain value:
        self.counter = 0

        self.buffer_counter = 0

    def run_study(self, q2):

        # wait for 30 seconds to simulate exploration
        time.sleep(30)

        vis_data = self.sample_episodes()

        # prepare logs:
        logs = {'num_frames': 2560,
                'episodes_done': 0,
                'return_per_episode': 0,
                'num_frames_per_episode': 0,
                'entropy': 0,
                'value': 0,
                'policy_loss': 0,
                'value_loss': 0,
                'loss': 0,
                'grad_norm': 0}

        # fk: integrate feedback into buffer:
        while not q2.empty():
            feedback, p_idx, e_idx, buffer_counter = q2.get()
            print('buffer_counter: ', buffer_counter)
            print('self.buffer_counter: ', self.buffer_counter)
            if buffer_counter == self.buffer_counter:
                self.integrate_feedback(feedback, p_idx, e_idx)

        self.counter += 1
        # whenever 14 minutes have passed, update the buffer file:
        if self.counter % 28 == 0:
            # save buffer to pickle file:
            # fk: generate file name with current model name and time stamp:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            file = f'buffer_with_feedback_{self.counter}_{timestamp}.pkl'
            with open(file, 'wb') as f:
                pickle.dump(self.proc_dfs, f)
            print('buffer saved to pickle file')

            # fk: increase buffer_counter by 1:
            self.buffer_counter += 1

            if self.counter < 140:
                print('updating buffer file')
                # create file name with current counter value:
                file = f'buffer_initialisation_{self.counter}.pkl'
                # get the corresponding buffer file and set self.proc_dfs to it:
                with open(file, 'rb') as f:
                    self.proc_dfs = pickle.load(f)
                # reset self.episodes_done:
                self.episodes_done = 0
                # count how many episodes in the buffer are done:
                for proc_df in self.proc_dfs:
                    self.episodes_done += proc_df[0].shape[0]
                    # if proc_df[1] == 0 subtract 1 from episodes_done (because the last episode is not done yet):
                    if not proc_df[1]:
                        self.episodes_done -= 1
                print('episodes done: ', self.episodes_done)

        return logs, vis_data, 0, 0, self.buffer_counter

    def update_parameters(self, q2):

        # fk: integrate feedback into buffer:
        while not q2.empty():
            feedback, p_idx, e_idx = q2.get()
            self.integrate_feedback(feedback, p_idx, e_idx)

        # Collect new experiences
        exps_new, logs, episodes_end_indexes = self.collect_new_experiences()
        '''
        exps is a DictList. The keys correspond to the column names of a process dataframe from proc_dfs (see above).
        exps.obs is a DictList with the following keys ['image', 'instr']
        exps.obj.image is a (n_procs * n_frames_per_proc) x image_size 4D tensor
        exps.obs.instr is a (n_procs * n_frames_per_proc) x (max number of words in an instruction) 2D tensor
        exps.memory is a (n_procs * n_frames_per_proc) x (memory_size = 2*image_embedding_size) 2D tensor
        exps.mask is (n_procs * n_frames_per_proc) x 1 2D tensor
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        '''

        '''
        fk: episodes_end_indexes is a list of lists of end indexes of episodes. Each list contains the end indexes of
        the episodes of one of the self.num_proc processes. We use this list to track when episodes are done.
        '''

        self.episodes_done += logs['episodes_done']

        # fk: if buffer is not empty use samples from buffer:
        # if self.buffer_usable:
        #     # fk: get random samples (of size self.num_frames_per_proc // 2 * self.num_procs from buffer for optimization:
        #     sampled_actions, sampled_values, sampled_masks, sampled_memories, sampled_rewards, sampled_orig_rewards, \
        #     sampled_obs, proc_inds, frame_inds, sampled_returnn = self.get_samples()
        #
        #     # fk: collect experiences with random samples:
        #     exps_old = self.collect_old_experiences(sampled_actions, sampled_values, sampled_masks, sampled_memories,
        #                                             sampled_rewards, sampled_orig_rewards, sampled_obs, sampled_returnn)
        # # fk: else use exps_new twice for optimization (only happens for the first training iteration)
        # else:
        #     exps_old = exps_new

        exps_old, a, b = self.collect_new_experiences()

        # fk: increase the length of the instructions in exps.old.obs.instr to match it with the length of the
        # instructions in exps.new.obs.instr (compare line 69 in babyai.utils.format.py):

        # fk: getting the length of the longest instruction:
        max_instr_len = max(len(exps_new.obs.instr[0]), len(exps_old.obs.instr[0]))

        instrs_old = torch.zeros((self.num_frames_per_proc // 2 * self.num_procs, max_instr_len), dtype=torch.int,
                                 device=self.device)
        instrs_new = torch.zeros((self.num_frames_per_proc // 2 * self.num_procs, max_instr_len), dtype=torch.int,
                                 device=self.device)

        for i, instr in enumerate(exps_old.obs.instr):
            instrs_old[i, :len(instr)] = instr
        for i, instr in enumerate(exps_new.obs.instr):
            instrs_new[i, :len(instr)] = instr

        # fk: re-assign:
        exps_old.obs.instr = instrs_old
        exps_new.obs.instr = instrs_new

        # fk: concatenate exps_new and exps_old to form one exps of size self.num_frames_per_proc * self.num_procs:
        exps = exps_new
        exps.obs = DictList(
            {'image': torch.cat((exps.obs.image, exps_old.obs.image), 0),
             'instr': torch.cat((exps.obs.instr, exps_old.obs.instr), 0)})
        exps.memory = torch.cat((exps.memory, exps_old.memory), 0)
        exps.mask = torch.cat((exps.mask, exps_old.mask), 0)
        exps.action = torch.cat((exps.action, exps_old.action), 0)
        exps.value = torch.cat((exps.value, exps_old.value), 0)
        exps.reward = torch.cat((exps.reward, exps_old.reward), 0)
        exps.orig_reward = torch.cat((exps.orig_reward, exps_old.orig_reward), 0)
        exps.advantage = torch.cat((exps.advantage, exps_old.advantage), 0)
        exps.returnn = torch.cat((exps.returnn, exps_old.returnn), 0)
        exps.log_prob = torch.cat((exps.log_prob, exps_old.log_prob), 0)
        exps.agent_x_pos = torch.cat((exps.agent_x_pos, exps_old.agent_x_pos), 0)
        exps.agent_y_pos = torch.cat((exps.agent_y_pos, exps_old.agent_y_pos), 0)
        exps.agent_dir = torch.cat((exps.agent_dir, exps_old.agent_dir), 0)
        exps.grid_width = torch.cat((exps.grid_width, exps_old.grid_width), 0)
        exps.grid_height = torch.cat((exps.grid_height, exps_old.grid_height), 0)
        exps.human_views = numpy.concatenate((exps.human_views, exps_old.human_views), 0)
        exps.missions = numpy.concatenate((exps.missions, exps_old.missions), 0)

        # fk: initialize arrays for stepwise policy_loss, value_loss, loss and entropy (the variables with '_acc'
        # suffix hold the values accumulated over all epochs):
        policy_loss_stepwise = numpy.zeros((self.num_frames_per_proc * self.num_procs,))
        policy_loss_stepwise_acc = numpy.zeros((self.num_frames_per_proc * self.num_procs,))
        value_loss_stepwise = numpy.zeros((self.num_frames_per_proc * self.num_procs,))
        value_loss_stepwise_acc = numpy.zeros((self.num_frames_per_proc * self.num_procs,))
        loss_stepwise = numpy.zeros((self.num_frames_per_proc * self.num_procs,))
        loss_stepwise_acc = numpy.zeros((self.num_frames_per_proc * self.num_procs,))
        entropy_stepwise = numpy.zeros((self.num_frames_per_proc * self.num_procs,))
        entropy_stepwise_acc = numpy.zeros((self.num_frames_per_proc * self.num_procs,))

        for _ in range(self.epochs):
            # Initialize log values

            print(f'epoch: {_}')

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []

            '''
            For each epoch, we create int(total_frames / batch_size + 1) batches, each of size batch_size (except
            maybe the last one. Each batch is divided into sub-batches of size recurrence (frames are contiguous in
            a sub-batch), but the position of each sub-batch in a batch and the position of each batch in the whole
            list of frames is random thanks to self._get_batches_starting_indexes().
            '''

            for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches (fk: in this case 2)
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss

                    model_results = self.acmodel(sb.obs, memory * sb.mask)
                    dist = model_results['dist']
                    value = model_results['value']
                    memory = model_results['memory']
                    extra_predictions = model_results['extra_predictions']

                    # fk: assign stepwise entropy of ith iteration:
                    entropy_stepwise[inds + i] = dist.entropy().detach().numpy()
                    entropy_stepwise_acc[inds + i] += dist.entropy().detach().numpy()

                    entropy = dist.entropy().mean()

                    '''fk: Calculation of policy_loss corresponds to section 3 in
                    [Schulman et al., 2015](https://arxiv.org/abs/1707.06347)'''
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    # fk: assign policy_loss_stepwise:
                    policy_loss_stepwise[inds + i] = - torch.min(surr1, surr2).detach().numpy()
                    policy_loss_stepwise_acc[inds + i] += - torch.min(surr1, surr2).detach().numpy()
                    policy_loss = -torch.min(surr1, surr2).mean()

                    '''fk: calculation of value_loss is not exactly the same as in
                    [Schulman et al., 2015](https://arxiv.org/abs/1707.06347); here, they take the same approach as
                    for policy_loss (for the same reason, i.e. to avoid too high adjustments)'''
                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    # fk: assign value_loss_stepwise:
                    value_loss_stepwise[inds + i] = torch.max(surr1, surr2).detach().numpy()
                    value_loss_stepwise_acc[inds + i] += torch.max(surr1, surr2).detach().numpy()
                    value_loss = torch.max(surr1, surr2).mean()

                    '''fk: the calculation of loss corresponds to formula (9) in
                    [Schulman et al., 2015](https://arxiv.org/abs/1707.06347)'''
                    # fk: assign loss_stepwise:
                    loss_stepwise[inds + i] = policy_loss_stepwise[inds + i] - self.entropy_coef * entropy_stepwise[
                        inds + i] + self.value_loss_coef * value_loss_stepwise[inds + i]
                    loss_stepwise_acc[inds + i] += policy_loss_stepwise[inds + i] - self.entropy_coef * \
                                                   entropy_stepwise[inds + i] + self.value_loss_coef * \
                                                   value_loss_stepwise[inds + i]
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(
                    p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())

        # fk: Calculate mean over all epochs of stepwise losses and entropy:
        policy_loss_stepwise_acc /= self.epochs
        value_loss_stepwise_acc /= self.epochs
        loss_stepwise_acc /= self.epochs
        entropy_stepwise_acc /= self.epochs

        # fk: attach stepwise losses and entropy to exps:
        exps.policy_loss_stepwise = policy_loss_stepwise_acc
        exps.value_loss_stepwise = value_loss_stepwise_acc
        exps.loss_stepwise = loss_stepwise_acc
        exps.entropy_stepwise = entropy_stepwise_acc

        # fk: separate exps again to either add them to the buffer (exps_new) or overwrite the buffer (exps_old):
        exps_new = exps[range(self.num_frames_per_proc // 2 * self.num_procs)]
        exps_old = exps[
            range(self.num_frames_per_proc // 2 * self.num_procs, self.num_frames_per_proc * self.num_procs)]

        # fk: overwrite buffer with updated old experiences:
        # if self.buffer_usable:
        #     self.overwrite_buffer(exps_old, proc_inds, frame_inds)

        # fk: populate buffer with new experiences:
        # self.populate_buffer(exps_new, episodes_end_indexes)

        vis_data = None  # self.sample_episodes()

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["loss"] = numpy.mean(log_losses)

        # save buffer to pickle file:
        # if self.buffer_usable:
        #     # fk: generate file name with current model name:
        #     file = f'buffer_initialisation.pkl'
        #     with open(file, 'wb') as f:
        #         pickle.dump(self.proc_dfs, f)
        #     print('buffer saved to pickle file')
        #     print(f'self.episodes_done: {self.episodes_done}')

        return logs, vis_data, self.amount_feedback_frames, self.episodes_done

    # function to randomly sample a list of episodes from the buffer:
    def sample_episodes(self):
        # fk: determine number of randomly selected episodes to visualize in frontend:
        episodes_amount = min(100, self.episodes_done)

        # fk: sorted list of indices of randomly selected episodes that shall be visualized in frontend (each episode
        # can be selected only once):
        selected_episodes = list(numpy.random.permutation(numpy.arange(0, self.episodes_done))[:episodes_amount])
        selected_episodes.sort()

        # fk: list of episodes that is later handed over to frontend through q (see train_rl.py):
        episode_list = []

        # fk: add the selected episodes to episode_list:
        for i in selected_episodes:
            p_idx = 0
            e_idx = i
            # if episode is not contained in process with index p_idx, go to next process:
            while e_idx >= self.proc_dfs[p_idx][0].shape[0] - (1 - self.proc_dfs[p_idx][1]):
                # subtract amount of episodes of current process:
                e_idx -= self.proc_dfs[p_idx][0].shape[0] - (1 - self.proc_dfs[p_idx][1])
                # go to next process:
                p_idx += 1

            # fk: pick episode:
            e = self.proc_dfs[p_idx][0].iloc[e_idx].copy()

            # fk: pass a minimum amount of data since the queue will get stuck otherwise (i.e. the update
            # interval of 15 seconds is too small.)
            e.loc['loss_episode_wise'] = round(sum(e.loc['loss_stepwise']) / len(e.loc['loss_stepwise']), 2)
            e.loc['reward_episode_wise'] = round(sum(e.loc['rewards']) / len(e.loc['rewards']) / 20 / 2, 2)
            e.loc['mission'] = e.loc['missions'][0]
            e = e.drop(
                labels=['obs_image', 'obs_instr', 'memories', 'masks', 'values', 'returnn',
                        # 'policy_loss_stepwise', 'loss_stepwise', 'value_loss_stepwise', 'advantages',
                        'rewards', 'missions'])

            # fk: appending process index and episode index to e (needed later to overwrite with human feedback):
            e.loc['proc_idx'] = p_idx
            e.loc['episode_idx'] = e_idx
            e.loc['buffer_idx'] = self.buffer_counter

            # fk: finally append episode to episode_list:
            episode_list.append(e)

        # fk: when no episode is finished yet, set vis_data to None (will be handled accordingly in train_rl.py):
        if not episode_list:
            vis_data = None
        else:
            # fk: sort episode_list by episode index:
            episode_list.sort(key=lambda x: x['episode_idx'])
            vis_data = pd.concat(episode_list, axis=1, ignore_index=True).T

        return vis_data

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """

        # fk: self.recurrence default is 20 (see arguments.py),
        # num_frames: int = self.num_frames_per_proc * self.num_procs -> 40 * 64 = 2560
        indexes = numpy.arange(0, self.num_frames * 2, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # fk: self.batch_size default is 1280 (see arguments.py), therefore num_indexes is 1280 // 20 = 64
        num_indexes = self.batch_size // self.recurrence

        batches_starting_indexes = [(indexes[i:i + num_indexes]) for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def get_samples(self):
        # instantiating lists to store the indexes of processes and frames while sampling frames for PPO optimization:
        proc_inds = []
        frame_inds = []

        # fk: instantiating the needed data structures for optimization; We sample num_frames_per_proc // 2 + 1 (!)
        # frames because the last mask/obs is needed to compute next_value in collect_old_experiences() in base.py:
        sampled_actions = torch.zeros(self.num_procs, self.num_frames_per_proc // 2 + 1, device=self.device)
        sampled_values = torch.zeros(self.num_procs, self.num_frames_per_proc // 2 + 1, device=self.device)
        sampled_masks = torch.zeros(self.num_procs, self.num_frames_per_proc // 2 + 1, device=self.device)
        sampled_memories = torch.zeros(self.num_procs, self.num_frames_per_proc // 2 + 1,
                                       256, device=self.device)  # fk: 256 comes from model.py lines 209-210
        sampled_rewards = torch.zeros(self.num_procs, self.num_frames_per_proc // 2 + 1, device=self.device)
        sampled_orig_rewards = torch.zeros(self.num_procs, self.num_frames_per_proc // 2 + 1, device=self.device)
        sampled_obs = [None] * self.num_procs
        sampled_returnn = torch.zeros(self.num_procs, self.num_frames_per_proc // 2 + 1, device=self.device)
        # also sample the 'feedback' column from the buffer:
        sampled_feedback = torch.zeros(self.num_procs, self.num_frames_per_proc // 2 + 1, device=self.device)

        # fk: randomly sample chunks of frames of size self.num_frames_per_proc // 2 + 1 from buffer:
        for i in range(self.num_procs):

            # fk: initialize data structures to temporarily store sampled data:
            sampled_actions_temp = []
            sampled_values_temp = []
            sampled_masks_temp = []
            sampled_memories_temp = []
            sampled_rewards_temp = []
            sampled_orig_rewards_temp = []
            sampled_obs_image_temp = []
            sampled_agent_dir_temp = []
            sampled_missions_temp = []
            sampled_returnn_temp = []
            sampled_feedback_temp = []

            # fk: select a process and starting frame:
            # if self.feedback_episodes is not empty take a random process from there:
            if self.feedback_episodes and i % 2 == 0:
                proc_idx, e_idx = random.choice(self.feedback_episodes)
                # fk: get episode:
                e = self.proc_dfs[proc_idx][0].iloc[e_idx].copy()
                # get one of the first len(e['actions']) - self.num_frames_per_proc // 2 - 1 frames of e:
                frame_idx = random.randint(0, len(e['actions']) - (self.num_frames_per_proc // 2 - 1))
                # add the amount of frames of all episodes that are before e to the frame_idx:
                frame_idx += sum([len(self.proc_dfs[proc_idx][0].iloc[i]['actions']) for i in range(e_idx)])
            # else take a random process from all processes:
            else:
                proc_idx = random.randint(0, self.num_procs - 1)

                # fk: we cannot use the last self.num_frames_per_proc // 2 frames of the buffer since we need the
                # self.num_frames_per_proc // 2 + 1th frame to compute next_value in collect_old_experiences() in
                # base.py
                usable_frames = self.frames_in_buffer - self.num_frames
                # fk: create list of all possible starting frames (every (self.num_frames_per_proc // 2)th frame is
                # possible):
                starting_frames = list(range(0, usable_frames // self.num_procs, self.num_frames_per_proc // 2))

                # fk: randomly select a frame:
                frame_idx = random.choice(starting_frames)

            # fk: store process index in list:
            proc_inds.append(proc_idx)
            # fk: store frame_idx in list:
            frame_inds.append(frame_idx)
            proc_df = self.proc_dfs[proc_idx][0]

            # fk: the amount of frames to add per sampled chunk:
            frames_to_add = self.num_frames_per_proc // 2 + 1

            # fk: finding episode to start with:
            e_idx = 0
            while len(proc_df.iloc[e_idx]['actions']) < frame_idx:
                # fk: subtract omitted frames from frame_idx
                frame_idx -= len(proc_df.iloc[e_idx]['actions'])
                # fk: jump to next episode
                e_idx += 1
            e = proc_df.iloc[e_idx].copy()

            # fk: retrieve necessary data:
            while frames_to_add > 0:
                sampled_actions_temp = sampled_actions_temp + e['actions'][frame_idx:min(frame_idx + frames_to_add,
                                                                                         len(e['actions']))]
                sampled_values_temp = sampled_values_temp + e['values'][frame_idx:min(frame_idx + frames_to_add,
                                                                                      len(e['values']))]
                sampled_masks_temp = sampled_masks_temp + e['masks'][frame_idx:min(frame_idx + frames_to_add,
                                                                                   len(e['masks']))]
                sampled_memories_temp = sampled_memories_temp + e['memories'][frame_idx:min(frame_idx + frames_to_add,
                                                                                            len(e['memories']))]
                sampled_rewards_temp = sampled_rewards_temp + e['rewards'][frame_idx:min(frame_idx + frames_to_add,
                                                                                         len(e['rewards']))]
                sampled_orig_rewards_temp = sampled_orig_rewards_temp + e['orig_rewards'][
                                                                        frame_idx:min(frame_idx + frames_to_add,
                                                                                      len(e['orig_rewards']))]
                sampled_obs_image_temp = sampled_obs_image_temp + e['obs_image'][
                                                                  frame_idx:min(frame_idx + frames_to_add,
                                                                                len(e['obs_image']))]
                sampled_agent_dir_temp = sampled_agent_dir_temp + e['agent_dir'][
                                                                  frame_idx:min(frame_idx + frames_to_add,
                                                                                len(e['agent_dir']))]
                sampled_missions_temp = sampled_missions_temp + e['missions'][
                                                                frame_idx:min(frame_idx + frames_to_add,
                                                                              len(e['missions']))]
                sampled_returnn_temp = sampled_returnn_temp + e['returnn'][frame_idx:min(frame_idx + frames_to_add,
                                                                                         len(e['returnn']))]
                sampled_feedback_temp = sampled_feedback_temp + e['feedback'][frame_idx:min(frame_idx + frames_to_add,
                                                                                            len(e['feedback']))]
                # fk: reduce number of frames that need to be added:
                frames_to_add -= len(e['actions']) - frame_idx

                # fk: reset frame_idx (only relevant if frames_to_add is still > 0):
                frame_idx = 0

                # fk: jump to next episode (only relevant if frames_to_add is still > 0):
                if proc_df.shape[0] - 1 > e_idx:
                    e_idx += 1
                else:
                    e_idx = 0
                e = proc_df.iloc[e_idx].copy()

            # fk: finally store ith sampled data in the corresponding data structures:
            sampled_actions[i] = torch.tensor(sampled_actions_temp, device=self.device)
            sampled_values[i] = torch.tensor(sampled_values_temp, device=self.device)
            sampled_masks[i] = torch.tensor(sampled_masks_temp, device=self.device)
            sampled_memories[i] = torch.tensor(numpy.asarray(sampled_memories_temp), device=self.device)
            sampled_rewards[i] = torch.tensor(sampled_rewards_temp, device=self.device)
            sampled_orig_rewards[i] = torch.tensor(sampled_orig_rewards_temp, device=self.device)
            sampled_obs_temp = []
            for j in range(self.num_frames_per_proc // 2 + 1):
                sampled_obs_temp.append({'image': sampled_obs_image_temp[j], 'direction': sampled_agent_dir_temp[j],
                                         'mission': sampled_missions_temp[j]})
            sampled_obs[i] = sampled_obs_temp
            sampled_returnn[i] = torch.tensor(sampled_returnn_temp, device=self.device)
            sampled_feedback[i] = torch.tensor(sampled_feedback_temp, device=self.device)

        # fk: transpose sampled data to be in correct format, i.e. ith row represents ith step of all processes:
        sampled_actions = sampled_actions.transpose(0, 1)
        sampled_values = sampled_values.transpose(0, 1)
        sampled_masks = sampled_masks.transpose(0, 1)
        sampled_memories = sampled_memories.transpose(0, 1)
        sampled_rewards = sampled_rewards.transpose(0, 1)
        sampled_orig_rewards = sampled_orig_rewards.transpose(0, 1)
        sampled_obs = numpy.array(sampled_obs).T.tolist()
        sampled_returnn = sampled_returnn.transpose(0, 1)
        sampled_feedback = sampled_feedback.transpose(0, 1)

        # add number of frames with feedback to self.amount_feedack_frames:
        self.amount_feedback_frames += int(sum(sum(sampled_feedback)))

        return sampled_actions, sampled_values, sampled_masks, sampled_memories, sampled_rewards, sampled_orig_rewards, \
               sampled_obs, proc_inds, frame_inds, sampled_returnn

    def populate_buffer(self, exps_new, episodes_end_indexes):
        """
        fk: Populates the buffer (proc_dfs) with the newly collected experiences.
        @param exps_new: the exps from collect_new_experiences method in base.py (after passing PPO)
        @param episodes_end_indexes: the end indexes of the episodes contained in exps_new
        """
        # fk: create and populate self.num_proc dataframes that hold episode data:
        for i, end_inds in enumerate(episodes_end_indexes):

            # fk: get data of ith process:
            proc_data = exps_new[range(i * self.num_frames_per_proc // 2,
                                       i * self.num_frames_per_proc // 2 + self.num_frames_per_proc // 2)]

            # fk: create dataframe that contains process data:
            df = pd.DataFrame()
            df['obs_image'] = [proc_data.obs.image[k].numpy().astype('uint8') for k in
                               range(proc_data.obs.image.shape[0])]
            df['obs_instr'] = [proc_data.obs.instr[k].numpy() for k in range(proc_data.obs.instr.shape[0])]
            df['memories'] = [proc_data.memory[k].numpy() for k in range(proc_data.memory.shape[0])]
            df['masks'] = proc_data.mask
            df['actions'] = proc_data.action
            df['values'] = proc_data.value
            df['rewards'] = proc_data.reward
            df['orig_rewards'] = proc_data.orig_reward
            df['advantages'] = proc_data.advantage
            df['returnn'] = proc_data.returnn
            df['agent_x_pos'] = proc_data.agent_x_pos
            df['agent_y_pos'] = proc_data.agent_y_pos
            df['agent_dir'] = proc_data.agent_dir
            df['grid_width'] = proc_data.grid_width
            df['grid_height'] = proc_data.grid_height
            df['human_views'] = proc_data.human_views
            df['missions'] = proc_data.missions
            df['entropy_stepwise'] = proc_data.entropy_stepwise
            df['policy_loss_stepwise'] = proc_data.policy_loss_stepwise
            df['value_loss_stepwise'] = proc_data.value_loss_stepwise
            df['loss_stepwise'] = proc_data.loss_stepwise

            # fk: initially, no human feedback has been provided to these episodes:
            df['feedback'] = [False] * (
                    self.num_frames_per_proc // 2)  # TODO: better: feedback-id -> +1 for each time given feedback

            # fk: initialize list of episode chunks
            episode_chunks = []

            # fk: if end_inds is empty then there is only one episode chunk:
            if not end_inds:
                episode_chunks.append(df)
            # fk: otherwise there are multiple episode chunks:
            else:
                # fk: transform end_inds into start_inds (allows easier processing in the next steps):
                start_inds = [x + 1 for x in end_inds]

                # fk: add 0 to start_inds:
                start_inds.insert(0, 0)

                # fk: if num_frames_per_proc // 2 is not contained in start_inds append it to create last episode chunk:
                if (self.num_frames_per_proc // 2) not in start_inds:
                    start_inds.append(self.num_frames_per_proc // 2)

                # fk: creating list of tuples of indices that form the bounds of an episode chunk:
                bounds = zip(start_inds, start_inds[1:])

                # fk: split dataframe into episode chunks:
                for (a, b) in bounds:
                    episode_df = df.iloc[a:b]
                    episode_chunks.append(episode_df)

            # fk: append episodes to process dataframe:
            # fk: if last row of proc_df[i] is not a full episode concatenate last row of proc_df[i] with episode chunk:
            if not self.proc_dfs[i][1]:
                for c in self.proc_dfs[i][0].columns:
                    self.proc_dfs[i][0].at[len(self.proc_dfs[i][0]) - 1, c] = \
                        self.proc_dfs[i][0].at[len(self.proc_dfs[i][0]) - 1, c] + episode_chunks[0][c].tolist()

                # fk: remove first episode chunk from episode_chunks for append operation:
                episode_chunks = episode_chunks[1:]

            # fk: the remaining chunks are appended to the corresponding process df:
            for episode_df in episode_chunks:
                new_row = episode_df.values.T.tolist()
                self.proc_dfs[i][0] = pd.concat(
                    [self.proc_dfs[i][0],
                     pd.Series(new_row, index=self.proc_dfs[i][0].columns[:len(new_row)]).to_frame().T],
                    ignore_index=True)

            # fk: set self.proc_dfs[i][1] to True or False depending on if the last row of proc_dfs[i][0] contains a
            # full episode (i.e. when end_inds contains self.num_frames_per_proc - 1):
            self.proc_dfs[i][1] = (self.num_frames_per_proc // 2 - 1) in end_inds

        # fk: update number of frames in buffer (needed for get_samples method):
        self.frames_in_buffer += self.num_frames

        # fk: set buffer_usable to True (gives hint that now we can get samples of old experiences from the buffer):
        if self.frames_in_buffer > self.num_frames:
            self.buffer_usable = True

    def overwrite_buffer(self, exps_old, proc_inds, frame_inds):
        """
        fk: overwrites old experience values of exps_old in the buffer.
        @param exps_old: the randomly sampled experiences from the buffer after being updated through
        collect_old_experiences (base.py) and PPO.
        @param proc_inds: the process indices that the samples contained in exps_old belong to
        @param frame_inds: the starting frames indices of the samples contained in exps_old
        """

        # fk: for each chunk of sampled frames overwrite the buffer with the updated data:
        for i, p_idx in enumerate(proc_inds):
            # fk get process:
            proc_df = self.proc_dfs[p_idx][0]

            # fk: finding episode to start with:
            frame_idx = frame_inds[i]
            e_idx = 0
            while len(proc_df.iloc[e_idx][
                          'actions']) < frame_idx:  # TODO: make more efficient by using episode indices from get_samples()
                # fk: subtract omitted frames from frame_idx
                frame_idx -= len(proc_df.iloc[e_idx]['actions'])
                # fk: jump to next episode
                e_idx += 1
            e = proc_df.iloc[e_idx].copy()

            # fk: the amount of frames that still need to be updated for this chunk of sampled frames:
            frames_to_overwrite = self.num_frames_per_proc // 2

            # fk: overwrite:
            while frames_to_overwrite > 0:
                # fk: get current values:
                policy_loss_stepwise = e['policy_loss_stepwise']
                value_loss_stepwise = e['value_loss_stepwise']
                loss_stepwise = e['loss_stepwise']
                entropy_stepwise = e['entropy_stepwise']

                # fk: find exps_old index to start with:
                start_idx = i * self.num_frames_per_proc // 2 + (self.num_frames_per_proc // 2 - frames_to_overwrite)

                # fk: find exps_old index to end with:
                end_idx = min(start_idx + frames_to_overwrite, start_idx + len(loss_stepwise) - frame_idx)

                # fk: overwrite current with new values:
                policy_loss_stepwise[frame_idx:min(frame_idx + frames_to_overwrite, len(policy_loss_stepwise))] = \
                    exps_old.policy_loss_stepwise[range(start_idx, end_idx)]
                value_loss_stepwise[frame_idx:min(frame_idx + frames_to_overwrite, len(value_loss_stepwise))] = \
                    exps_old.value_loss_stepwise[range(start_idx, end_idx)]
                loss_stepwise[frame_idx:min(frame_idx + frames_to_overwrite, len(loss_stepwise))] = \
                    exps_old.loss_stepwise[range(start_idx, end_idx)]
                entropy_stepwise[frame_idx:min(frame_idx + frames_to_overwrite, len(entropy_stepwise))] = \
                    exps_old.entropy_stepwise[range(start_idx, end_idx)]

                self.proc_dfs[p_idx][0].iloc[e_idx]['policy_loss_stepwise'] = policy_loss_stepwise
                self.proc_dfs[p_idx][0].iloc[e_idx]['value_loss_stepwise'] = value_loss_stepwise
                self.proc_dfs[p_idx][0].iloc[e_idx]['loss_stepwise'] = loss_stepwise
                self.proc_dfs[p_idx][0].iloc[e_idx]['entropy_stepwise'] = entropy_stepwise

                # fk: reduce number of frames that need to be overwritten:
                frames_to_overwrite -= len(e['loss_stepwise']) - frame_idx

                # fk: reset frame_idx (only relevant if frames_to_add is still > 0):
                frame_idx = 0

                # fk: jump to next episode (only relevant if frames_to_add is still > 0):
                if proc_df.shape[0] - 1 > e_idx:
                    e_idx += 1
                else:
                    e_idx = 0
                e = proc_df.iloc[e_idx].copy()

    def integrate_feedback(self, feedback, p_idx, e_idx):
        """
        fk: method to integrate human feedback into the buffer (i.e. proc_dfs)
        @param feedback: list of sentiment scores derived from the feedback provided by the human (ranging from -1.0 to
        1.0).
        @param p_idx: the index of the process that contains the episode that has been given feedback to.
        @param e_idx: the index of the episode that has been given feedback to
        """

        print(f'raw feedback: {feedback}')

        # fk: 'subtraction': after Meeting on 27.04.23 Yannick suggested to include this line:
        result = []
        for i in range(1, len(feedback)):
            diff = feedback[i] - feedback[i - 1]
            result.append(diff)

        feedback = [feedback[0]] + result

        print(f'feedback after subtraction: {feedback}')

        # turn every 1 in feedback into 0.1 and every -1 into -0.1 but keep the 0s as they are:
        # feedback = [0.1 if f == 1 else -0.1 if f == -1 else f for f in feedback]
        #
        # print(f'feedback after adjustments: {feedback}')

        # fk: the originally provided rewards (from base.py; values between 0.0 and 1.0)
        orig_rewards = torch.tensor(self.proc_dfs[p_idx][0].iloc[e_idx]['orig_rewards'], device=self.device)

        # fk: add feedback to original rewards:
        reshaped_rewards = orig_rewards + torch.tensor(feedback, device=self.device)

        print(f'orig rewards + feedback: {reshaped_rewards}')

        # fk: use -0.5 as lower bound:
        reshaped_rewards.map_(reshaped_rewards, lambda a, b: -0.5 if a < -0.5 else a)
        # fk: use 2.0 as upper bound:
        reshaped_rewards.map_(reshaped_rewards, lambda a, b: 1.0 if a > 1 else a)

        print(f'orig rewards + feedback clipped: {reshaped_rewards}')

        # fk: apply reshaping function provided in train_rl.py (reshape_reward)
        reshaped_rewards = self.reshape_reward(None, None, reshaped_rewards, None)

        print(f'reshaped orig rewards + feedback clipped: {reshaped_rewards}')

        # fk: overwrite buffer with updated reward values:
        self.proc_dfs[p_idx][0].iloc[e_idx]['rewards'] = reshaped_rewards.tolist()
        # fk: update the feedback column by setting all values to True (i.e. feedback has been integrated):
        self.proc_dfs[p_idx][0].iloc[e_idx]['feedback'] = [True] * len(reshaped_rewards.tolist())

        # fk: append p_idx and e_idx to feedbacks list if feedback has more than self.num_frames_per_proc // 2 frames-1:
        if len(feedback) > self.num_frames_per_proc // 2 - 1:
            self.feedback_episodes.append((p_idx, e_idx))
