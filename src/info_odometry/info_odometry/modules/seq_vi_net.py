import pdb
import torch
from torch import nn
from torch.nn import functional as F
from .info_model import flownet_featsize
from .pose_model import PoseModel


class SeqVINet(nn.Module):
    def __init__(self, args, belief_size, state_size, hidden_size, embedding_size, use_imu, activation_function='relu'):
        """
        (1) always use image (2) optionally use imu

        Input:
        -> belief_size: used for fusion rnn (type determined by belief_rnn)
        -> state_size: output size for feed into pose model
        -> hidden_size: use for fc embedding
        -> embedding_size: use for imu rnn
        """
        super().__init__()
        self.args = args
        self.use_imu = use_imu
        self.use_soft = args.soft
        self.use_hard = args.hard
        self.embedding_size = embedding_size
        self.act_fn = getattr(F, activation_function)
        self.dropout = getattr(F, 'dropout')
        if self.use_imu:
            if args.imu_rnn == 'lstm':
                self.rnn_embed_imu = nn.LSTM(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True,
                                             dropout=0.4)
            elif args.imu_rnn == 'gru':
                self.rnn_embed_imu = nn.GRU(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True,
                                            dropout=0.4)
            self.fc_embed_sensors = nn.Linear(2 * embedding_size, belief_size)
        else:
            self.fc_embed_sensors = nn.Linear(embedding_size, belief_size)
        if args.belief_rnn == 'lstm':
            self.rnn_fusion = nn.LSTM(input_size=belief_size, hidden_size=belief_size, num_layers=2, batch_first=True,
                                      dropout=0.4)
        elif args.belief_rnn == 'gru':
            self.rnn_fusion = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_fusion = nn.Linear(belief_size, hidden_size)
        self.fc_out_fusion = nn.Linear(hidden_size, state_size)

        if self.use_soft and self.use_imu:
            self.sigmoid = nn.Sigmoid()
            self.soft_fc_img = nn.Linear(2 * embedding_size, embedding_size)
            self.soft_fc_imu = nn.Linear(2 * embedding_size, embedding_size)

        if self.use_hard and self.use_imu:
            self.sigmoid = nn.Sigmoid()
            self.hard_fc_img = nn.Linear(2 * embedding_size, embedding_size)
            self.hard_fc_imu = nn.Linear(2 * embedding_size, embedding_size)
            if args.hard_mode == 'onehot':
                self.onehot_hard = True
            elif args.hard_mode == 'gumbel_soft':
                self.onehot_hard = False
            self.eps = 1e-10

        self.init_weights()

    def init_weights(self):
        """
        follow a deepvo pytorch implementation gitub repo
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                nn.init.xavier_normal_(m.weight.data)

            elif isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # if m.bias is not None:
                #     m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LSTMCell):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        # nn.init.orthogonal(param)
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        # Forget gate bias trick: Initially during training, it is often helpful
                        # to initialize the forget gate bias to a large value, to help information
                        # flow over longer time steps.
                        # In a PyTorch LSTM, the biases are stored in the following order:
                        # [ b_ig | b_fg | b_gg | b_og ]
                        # where, b_ig is the bias for the input gate,
                        # b_fg is the bias for the forget gate,
                        # b_gg (see LSTM docs, Variables section),
                        # b_og is the bias for the output gate.
                        # So, we compute the location of the forget gate bias terms as the
                        # middle one-fourth of the bias vector, and initialize them.

                        # First initialize all biases to zero
                        # nn.init.uniform_(param)
                        nn.init.constant_(param, 0.)
                        bias = getattr(m, name)
                        n = bias.size(0)
                        start, end = n // 4, n // 2
                        bias.data[start:end].fill_(10.)

    # Operates over (previous) state, (previous) poses, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # p : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    # @jit.script_method
    def step(self,
             prev_state,
             poses,
             prev_belief,
             observations,
             rnn_embed_imu_hiddens,
             fusion_lstm_hiddens,
             gumbel_temperature=0.5):
        """
        prev_state: not used (for code consistency in main.py)
        prev_belief: i.e. prev_hidden (for code consistency in main.py)
        gumbel_temperature: the default value 0.5 is used for evaluation
        """
        if self.use_imu:
            observations_visual = observations[0].squeeze(0)  # [batch, 1024]
            observations_imu = observations[1]  # [batch, 11, 6]

        fusion_features = prev_belief
        #fusion_hiddens, fusion_features, out_features = prev_belief, prev_belief, torch.empty(0)

        # if self.args.belief_rnn == 'lstm':
        #     fusion_lstm_hiddens = (prev_belief.unsqueeze(0).repeat(2, 1, 1),
        #                            prev_belief.unsqueeze(0).repeat(2, 1, 1))

        # if self.use_imu:
        #     rnn_embed_imu_hiddens = [(torch.empty(0))]
        #     prev_rnn_embed_imu_hidden = torch.zeros(2, 1, self.args.embedding_size, device=self.args.device)
        #     if self.args.imu_rnn == 'lstm':
        #         rnn_embed_imu_hiddens[0] = (prev_rnn_embed_imu_hidden, prev_rnn_embed_imu_hidden)
        #     elif self.args.imu_rnn == 'gru':
        #         rnn_embed_imu_hiddens[0] = prev_rnn_embed_imu_hidden

        if self.use_imu:
            hidden, rnn_embed_imu_hiddens = self.rnn_embed_imu(observations_imu,
                                                               rnn_embed_imu_hiddens)

            fused_feat = torch.cat([observations_visual, hidden[:,-1,:]], dim=1)

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

            hidden = self.act_fn(self.dropout(self.fc_embed_sensors(fused_feat), 0.5))
        else:
            hidden = self.act_fn(self.dropout(self.fc_embed_sensors(observations), 0.5))

        if self.args.belief_rnn == 'gru':
            fusion_features = self.rnn_fusion(hidden, fusion_features)
        elif self.args.belief_rnn == 'lstm':
            hidden = hidden.unsqueeze(1)
            fusion_feature_rnn, fusion_lstm_hiddens = self.rnn_fusion(hidden, fusion_lstm_hiddens)
            fusion_features = fusion_feature_rnn.squeeze(1)

        hidden = self.act_fn(self.dropout(self.fc_embed_fusion(fusion_features), 0.6))
        out_features = self.fc_out_fusion(hidden)

        hidden = [fusion_features,
                  out_features,
                  rnn_embed_imu_hiddens,
                  fusion_lstm_hiddens]

        return hidden

    # Operates over (previous) state, (previous) poses, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
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
        """
        prev_state: not used (for code consistency in main.py)
        prev_belief: i.e. prev_hidden (for code consistency in main.py)
        gumbel_temperature: the default value 0.5 is used for evaluation
        """
        if self.use_imu:
            observations_visual = observations[0]  # [batch, 1024]
            observations_imu = observations[1]  # [batch, 11, 6]
        use_pose_model = True if type(poses) == PoseModel else False
        T = self.args.clip_length + 1
        fusion_hiddens, fusion_features, out_features = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * (
                    T - 1)
        fusion_hiddens[0], fusion_features[0] = prev_belief, prev_belief
        if self.args.belief_rnn == 'lstm':
            fusion_lstm_hiddens = fusion_hiddens = [torch.empty(0)] * T
            fusion_lstm_hiddens[0] = (
            prev_belief.unsqueeze(0).repeat(2, 1, 1), prev_belief.unsqueeze(0).repeat(2, 1, 1))

        if self.use_imu:
            running_batch_size = prev_belief.size()[0]
            rnn_embed_imu_hiddens = [(torch.empty(0))] * T
            prev_rnn_embed_imu_hidden = torch.zeros(2, running_batch_size, self.args.embedding_size,
                                                    device=self.args.device)
            if self.args.imu_rnn == 'lstm':
                rnn_embed_imu_hiddens[0] = (prev_rnn_embed_imu_hidden, prev_rnn_embed_imu_hidden)
            elif self.args.imu_rnn == 'gru':
                rnn_embed_imu_hiddens[0] = prev_rnn_embed_imu_hidden

        if use_pose_model:
            pred_poses = [torch.empty(0)] * (T - 1)

        for t in range(T - 1):
            t_ = t - 1  # Use t_ to deal with different time indexing for observations

            if self.use_imu:
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

                hidden = self.act_fn(self.dropout(self.fc_embed_sensors(fused_feat), 0.5))
            else:
                hidden = self.act_fn(self.dropout(self.fc_embed_sensors(observations[t_ + 1]), 0.5))

            if self.args.belief_rnn == 'gru':
                fusion_features[t + 1] = self.rnn_fusion(hidden, fusion_features[t])
            elif self.args.belief_rnn == 'lstm':
                hidden = hidden.unsqueeze(1)
                fusion_feature_rnn, fusion_lstm_hiddens[t + 1] = self.rnn_fusion(hidden, fusion_lstm_hiddens[t])
                fusion_features[t + 1] = fusion_feature_rnn.squeeze(1)
            hidden = self.act_fn(self.dropout(self.fc_embed_fusion(fusion_features[t + 1]), 0.6))
            out_features[t_ + 1] = self.fc_out_fusion(hidden)
            if use_pose_model:
                with torch.no_grad():
                    pred_poses[t_ + 1] = poses(out_features[t_ + 1])

        hidden = [torch.stack(fusion_features, dim=0), None, None, None, torch.stack(out_features, dim=0), None, None]
        if use_pose_model:
            hidden += [torch.stack(pred_poses, dim=0)]
            if self.args.eval_uncertainty: hidden += [None]
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
