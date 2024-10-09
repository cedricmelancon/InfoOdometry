import torch
from torch import nn
from torch.nn import functional as F


class DoubleHiddenVITransitionModel(nn.Module):
    def __init__(self, args, belief_size, state_size, hidden_size, embedding_size, activation_function='relu',
                 min_std_dev=0.1):
        super().__init__()
        self.args = args
        self.use_soft = args.soft
        self.use_hard = args.hard
        self.embedding_size = embedding_size
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_posterior = nn.Linear(2 * state_size + 2 * embedding_size, belief_size)
        self.fc_embed_state_prior = nn.Linear(2 * state_size + 6 * args.pose_tiles, belief_size)
        if args.belief_rnn == 'lstm':
            self.rnn_posterior = nn.LSTM(input_size=belief_size, hidden_size=belief_size, num_layers=2,
                                         batch_first=True)
            self.rnn_prior = nn.LSTM(input_size=belief_size, hidden_size=belief_size, num_layers=2, batch_first=True)
        elif args.belief_rnn == 'gru':
            self.rnn_posterior = nn.GRUCell(belief_size, belief_size)
            self.rnn_prior = nn.GRUCell(belief_size, belief_size)
        if args.imu_rnn == 'lstm':
            self.rnn_embed_imu = nn.LSTM(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True)
        elif args.imu_rnn == 'gru':
            self.rnn_embed_imu = nn.GRU(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True)
        self.fc_embed_belief_posterior = nn.Linear(belief_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)

        if self.use_soft:
            self.sigmoid = nn.Sigmoid()
            self.soft_fc_img = nn.Linear(2 * embedding_size, embedding_size)
            self.soft_fc_imu = nn.Linear(2 * embedding_size, embedding_size)

        if self.use_hard:
            self.sigmoid = nn.Sigmoid()
            self.hard_fc_img = nn.Linear(2 * embedding_size, embedding_size)
            self.hard_fc_imu = nn.Linear(2 * embedding_size, embedding_size)
            if args.hard_mode == 'onehot':
                self.onehot_hard = True
            elif args.hard_mode == 'gumbel_soft':
                self.onehot_hard = False
            self.eps = 1e-10

    # Operates over (previous) state, (previous) poses, (previous) belief, (previous) nonterminals (mask),
    # and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state
    # that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # p : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # @jit.script_method
    def forward(self, prev_state, poses, prev_belief, observations, gumbel_temperature=0.5):
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with
        # inplace writes)
        observations_visual = observations[0]
        observations_imu = observations[1]
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
        if self.args.belief_rnn == 'lstm':
            prior_lstm_hiddens, posterior_lstm_hiddens = ([(torch.empty(0), torch.empty(0))] * T,
                                                          [(torch.empty(0), torch.empty(0))] * T)
            prior_lstm_hiddens[0] = (prev_belief.unsqueeze(0).repeat(2, 1, 1), prev_belief.unsqueeze(0).repeat(2, 1, 1))
            posterior_lstm_hiddens[0] = (prev_belief.unsqueeze(0).repeat(2, 1, 1),
                                         prev_belief.unsqueeze(0).repeat(2, 1, 1))
        if use_pose_model:
            pred_poses, pred_stds = [torch.empty(0)] * (T - 1), [torch.empty(0)] * (T - 1)
            # pred_poses, pred_pose_errs = [torch.empty(0)] * (T-1), [torch.empty(0)] * (T-1)

        running_batch_size = prev_belief.size()[0]
        rnn_embed_imu_hiddens = [torch.empty(0)] * T
        prev_rnn_embed_imu_hidden = torch.zeros(2,
                                                running_batch_size,
                                                self.args.embedding_size,
                                                device=self.args.device)
        if self.args.imu_rnn == 'lstm':
            rnn_embed_imu_hiddens[0] = (prev_rnn_embed_imu_hidden, prev_rnn_embed_imu_hidden)
        elif self.args.imu_rnn == 'gru':
            rnn_embed_imu_hiddens[0] = prev_rnn_embed_imu_hidden

        # Loop over time sequence
        for t in range(T - 1):
            t_ = t - 1  # Use t_ to deal with different time indexing for poses and observations
            # Update posterior_states
            hidden, rnn_embed_imu_hiddens[t + 1] = self.rnn_embed_imu(observations_imu[t_ + 1],
                                                                      rnn_embed_imu_hiddens[t])
            fused_feat = torch.cat([observations_visual[t_ + 1], hidden[:, -1, :]], dim=1)
            if self.use_soft:
                soft_mask_img = self.sigmoid(self.soft_fc_img(fused_feat))
                soft_mask_imu = self.sigmoid(self.soft_fc_imu(fused_feat))
                soft_mask = torch.ones_like(fused_feat).to(device=self.args.device)
                soft_mask[:, :self.embedding_size] = soft_mask_img
                soft_mask[:, self.embedding_size:] = soft_mask_imu
                fused_feat = fused_feat * soft_mask
            if self.use_hard:
                prob_img = self.sigmoid(self.hard_fc_img(fused_feat))
                prob_imu = self.sigmoid(self.hard_fc_imu(fused_feat))
                hard_mask_img = self.gumbel_sigmoid(prob_img, gumbel_temperature)
                hard_mask_imu = self.gumbel_sigmoid(prob_imu, gumbel_temperature)
                hard_mask_img = hard_mask_img[:, :, 0]
                hard_mask_imu = hard_mask_imu[:, :, 0]
                hard_mask = torch.ones_like(fused_feat).to(device=self.args.device)
                hard_mask[:, :self.embedding_size] = hard_mask_img
                hard_mask[:, self.embedding_size:] = hard_mask_imu
                fused_feat = fused_feat * hard_mask

            hidden = self.act_fn(
                self.fc_embed_state_posterior(torch.cat([fused_feat, posterior_states[t], prior_states[t]], dim=1)))
            if self.args.belief_rnn == 'gru':
                posterior_beliefs[t + 1] = self.rnn_posterior(hidden, posterior_beliefs[t])
            elif self.args.belief_rnn == 'lstm':
                hidden = hidden.unsqueeze(1)
                belief_rnn, posterior_lstm_hiddens[t + 1] = self.rnn_posterior(hidden, posterior_lstm_hiddens[t])
                posterior_beliefs[t + 1] = belief_rnn.squeeze(1)

            hidden = self.act_fn(self.fc_embed_belief_posterior(posterior_beliefs[t + 1]))
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
            hidden = self.act_fn(
                self.fc_embed_state_prior(torch.cat([_pose, posterior_states[t], prior_states[t]], dim=1)))
            if self.args.belief_rnn == 'gru':
                prior_beliefs[t + 1] = self.rnn_prior(hidden, prior_beliefs[t])
            elif self.args.belief_rnn == 'lstm':
                hidden = hidden.unsqueeze(1)
                belief_rnn, prior_lstm_hiddens[t + 1] = self.rnn_prior(hidden, prior_lstm_hiddens[t])
                prior_beliefs[t + 1] = belief_rnn.squeeze(1)

            hidden = self.act_fn(self.fc_embed_belief_prior(prior_beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])

            # Return new hidden states (init states are removed)
        if self.args.rec_type == 'posterior':
            hidden = [(torch.stack(posterior_beliefs[1:], dim=0), torch.stack(posterior_beliefs[1:], dim=0))]
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

    def gumbel_sigmoid(self, probs, tau):
        """
        input:
        -> probs: [batch, feat_size]: each element is the probability to be 1
        return:
        -> gumbel_dist: [batch, feat_size, 2]
            -> if self.onehot_hard == True:  one_hot vector (as in SelectiveFusion)
            -> if self.onehot_hard == False: gumbel softmax approx
        """
        log_probs = torch.stack((torch.log(probs + self.eps), torch.log(1 - probs + self.eps)),
                                dim=-1)  # [batch, feat_size, 2]
        gumbel = torch.rand_like(log_probs).to(device=self.args.device)
        gumbel = -torch.log(-torch.log(gumbel + self.eps) + self.eps)
        log_probs = log_probs + gumbel  # [batch, feat_size, 2]
        gumbel_dist = F.softmax(log_probs / tau, dim=-1)  # [batch, feat_size, 2]
        if self.onehot_hard:
            _shape = gumbel_dist.shape
            _, ind = gumbel_dist.max(dim=-1)
            gumbel_hard = torch.zeros_like(gumbel_dist).view(-1, _shape[-1])
            gumbel_hard.scatter_(dim=-1, index=ind.view(-1, 1), value=1.0)
            gumbel_hard = gumbel_hard.view(*_shape)
            gumbel_dist = (gumbel_hard - gumbel_dist).detach() + gumbel_dist
        return gumbel_dist
