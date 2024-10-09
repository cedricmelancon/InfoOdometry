import torch
from torch import nn
from torch.nn import functional as F


class DoubleStochasticTransitionModel(nn.Module):
    def __init__(self, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu',
                 min_std_dev=0.1):
        super().__init__()
        self.args = args
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev  # 6 * args.pose_tiles + embedding_size
        self.stochastic_mode = args.stochastic_mode
        if args.belief_rnn == 'lstm': raise NotImplementedError('do not support lstm for double-stochastic')

        if self.stochastic_mode == 'v1':
            self.fc_embed_state_posterior = nn.Linear(embedding_size, belief_size)
            self.fc_embed_state_prior = nn.Linear(6 * args.pose_tiles, belief_size)
            self.rnn_posterior = nn.GRUCell(belief_size, 2 * state_size)
            self.rnn_prior = nn.GRUCell(belief_size, 2 * state_size)
        elif self.stochastic_mode == 'v2':
            self.fc_embed_state_posterior = nn.Linear(state_size + embedding_size, belief_size)
            self.fc_embed_state_prior = nn.Linear(state_size + 6 * args.pose_tiles, belief_size)
            self.rnn_posterior = nn.GRUCell(belief_size, 2 * state_size)
            self.rnn_prior = nn.GRUCell(belief_size, 2 * state_size)
        elif self.stochastic_mode == 'v3':
            self.fc_embed_state_posterior = nn.Linear(state_size + embedding_size, belief_size)
            self.fc_embed_state_prior = nn.Linear(state_size + 6 * args.pose_tiles, belief_size)
            self.rnn_posterior = nn.GRUCell(belief_size, state_size)
            self.rnn_prior = nn.GRUCell(belief_size, state_size)
            self.fc_embed_rnn_posterior = nn.Linear(state_size, hidden_size)
            self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
            self.fc_embed_rnn_prior = nn.Linear(state_size, hidden_size)
            self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)

    # Operates over (previous) state, (previous) poses, (previous) belief, (previous) nonterminals (mask),
    # and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that
    # gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # p : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # @jit.script_method
    def forward(self, prev_state, poses, prev_belief, observations):
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        use_pose_model = True if type(poses) == PoseModel else False  # type(poses) == torch.Tensor # [time, batch, 6]
        T = self.args.clip_length + 1  # number of time steps # self.args.clip_length = poses.size(0)
        (prior_beliefs,
         posterior_beliefs,
         prior_states,
         prior_means,
         prior_std_devs,
         posterior_states,
         posterior_means,
         posterior_std_devs) = ([torch.empty(0)] * T,
                                [torch.empty(0)] * T,
                                [torch.empty(0)] * T,
                                [torch.empty(0)] * T,
                                [torch.empty(0)] * T,
                                [torch.empty(0)] * T,
                                [torch.empty(0)] * T,
                                [torch.empty(0)] * T)
        (prior_beliefs[0],
         posterior_beliefs[0],
         prior_states[0],
         posterior_states[0]) = prev_belief, prev_belief, prev_state, prev_state
        if use_pose_model:
            pred_poses, pred_stds = [torch.empty(0)] * (T - 1), [torch.empty(0)] * (T - 1)
            # pred_poses, pred_pose_errs = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)
        # Loop over time sequence
        for t in range(T - 1):
            t_ = t - 1  # Use t_ to deal with different time indexing for poses and observations
            # Update posterior_states
            if self.stochastic_mode == 'v1':
                hidden = self.act_fn(self.fc_embed_state_posterior(observations[t_ + 1]))
                posterior_beliefs[t + 1] = self.rnn_posterior(hidden,
                                                              torch.cat([posterior_states[t], prior_states[t]], dim=1))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(posterior_beliefs[t + 1], 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(
                    posterior_means[t + 1])
            elif self.stochastic_mode == 'v2':
                hidden = self.act_fn(
                    self.fc_embed_state_posterior(torch.cat([observations[t_ + 1], prior_states[t]], dim=1)))
                posterior_beliefs[t + 1] = self.rnn_posterior(hidden,
                                                              torch.cat([posterior_states[t], posterior_states[t]],
                                                                        dim=1))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(posterior_beliefs[t + 1], 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(
                    posterior_means[t + 1])
            elif self.stochastic_mode == 'v3':
                hidden = self.act_fn(
                    self.fc_embed_state_posterior(torch.cat([observations[t_ + 1], prior_states[t]], dim=1)))
                posterior_beliefs[t + 1] = self.rnn_posterior(hidden, posterior_states[t])
                hidden = self.act_fn(self.fc_embed_rnn_posterior(posterior_beliefs[t + 1]))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(
                    posterior_means[t + 1])

            # Get poses: use pose model in evaluation where no gt poses are available
            if use_pose_model:
                with torch.no_grad():
                    _pose = poses(posterior_means[t + 1])
                    if self.args.eval_uncertainty:
                        _plist = []
                        for k in range(100):
                            _plist.append(poses(posterior_means[t + 1] +
                                                posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])))
                        _plist = torch.stack(_plist, dim=0)  # [k, batch, 6]
                        # _pose = _plist.mean(dim=0)
                        pred_stds[t_ + 1] = torch.std(torch.norm(_plist, p=2, dim=2), dim=0)
                    pred_poses[t_ + 1] = _pose
            else:
                _pose = poses[t_ + 1]
            _pose = _pose.repeat(1, self.args.pose_tiles)

            # Update state_priors
            if self.stochastic_mode == 'v1':
                hidden = self.act_fn(self.fc_embed_state_prior(_pose))
                prior_beliefs[t + 1] = self.rnn_prior(hidden, torch.cat([posterior_states[t], prior_states[t]], dim=1))
                prior_means[t + 1], _prior_std_dev = torch.chunk(prior_beliefs[t + 1], 2, dim=1)
                prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
                prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            elif self.stochastic_mode == 'v2':
                hidden = self.act_fn(self.fc_embed_state_prior(torch.cat([_pose, posterior_states[t]], dim=1)))
                prior_beliefs[t + 1] = self.rnn_prior(hidden, torch.cat([prior_states[t], prior_states[t]], dim=1))
                prior_means[t + 1], _prior_std_dev = torch.chunk(prior_beliefs[t + 1], 2, dim=1)
                prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
                prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            elif self.stochastic_mode == 'v3':
                hidden = self.act_fn(self.fc_embed_state_prior(torch.cat([_pose, posterior_states[t]], dim=1)))
                prior_beliefs[t + 1] = self.rnn_prior(hidden, prior_states[t])
                hidden = self.act_fn(self.fc_embed_rnn_prior(prior_beliefs[t + 1]))
                prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
                prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
                prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])

                # Return new hidden states (init states are removed)
        if self.args.rec_type == 'posterior':
            hidden = [torch.stack(posterior_beliefs[1:], dim=0)]
        elif self.args.rec_type == 'prior':
            hidden = [torch.stack(prior_beliefs[1:], dim=0)]
        hidden += [torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0),
                   torch.stack(prior_std_devs[1:], dim=0)]
        hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0),
                   torch.stack(posterior_std_devs[1:], dim=0)]
        if use_pose_model:
            hidden += [torch.stack(pred_poses, dim=0)]
            if self.args.eval_uncertainty or self.args.eval_failure:
                hidden += [torch.stack(pred_stds, dim=0)]
        return hidden