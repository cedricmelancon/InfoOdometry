import torch
from torch import nn
from torch.nn import functional as F
from info_odometry.info_model import PoseModel
import collections
import numpy as np

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
        self.T = self.args.clip_length + 1
        
        self.embedding_size = embedding_size
        self.act_fn = getattr(F, activation_function)
        self.dropout = getattr(F, 'dropout')
        if self.use_imu:
            if args.imu_rnn == 'lstm':
                self.rnn_embed_imu = nn.LSTM(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True, dropout=0.4)
            elif args.imu_rnn == 'gru':
                self.rnn_embed_imu = nn.GRU(input_size=6, hidden_size=embedding_size, num_layers=2, batch_first=True, dropout=0.4)
            self.fc_embed_sensors = nn.Linear(2 * embedding_size, belief_size)
        else:
            self.fc_embed_sensors = nn.Linear(embedding_size, belief_size)
        if args.belief_rnn == 'lstm':
            self.rnn_fusion = nn.LSTM(input_size=belief_size, hidden_size=belief_size, num_layers=2, batch_first=True, dropout=0.4)
        elif args.belief_rnn == 'gru':
            self.rnn_fusion = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_fusion = nn.Linear(belief_size, hidden_size)
        self.fc_out_fusion = nn.Linear(hidden_size, state_size)

        self.init_weights()

        self.rnn_embed_imu_hiddens = collections.deque(maxlen=self.args.clip_length + 1)
        self.fusion_lstm_hiddens = collections.deque(maxlen=self.args.clip_length + 1)
        
        self.out_features = collections.deque(maxlen=self.args.clip_length)
        self.pred_pose = collections.deque(maxlen=self.args.clip_length)

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


    def execute_model(self, 
                      rnn_embed_imu_hiddens, 
                      observations_imu, 
                      observations_visual, 
                      observation, 
                      fusion_features, 
                      fusion_lstm_hiddens,
                      poses,
                      t):
        use_pose_model = True if type(poses) == PoseModel else False
        t_ = t - 1 # Use t_ to deal with different time indexing for observations

        if self.use_imu:
            hidden, rnn_embed_imu_hiddens[t + 1] = self.rnn_embed_imu(observations_imu[t_ + 1], rnn_embed_imu_hiddens[t])
            fused_feat = torch.cat([observations_visual[t_ + 1], hidden[:,-1,:]], dim=1)
                
            hidden = self.act_fn(self.dropout(self.fc_embed_sensors(fused_feat), 0.5))
        else:
            hidden = self.act_fn(self.dropout(self.fc_embed_sensors(observation), 0.5))
            
        if self.args.belief_rnn == 'gru':
            fusion_features[t + 1] = self.rnn_fusion(hidden, fusion_features[t])
        elif self.args.belief_rnn == 'lstm':
            hidden = hidden.unsqueeze(1)
            fusion_feature_rnn, fusion_lstm_hiddens[t + 1] = self.rnn_fusion(hidden, fusion_lstm_hiddens[t])
            fusion_features[t + 1] = fusion_feature_rnn.squeeze(1)
        hidden = self.act_fn(self.dropout(self.fc_embed_fusion(fusion_features[t + 1]), 0.6))
        out_features = self.fc_out_fusion(hidden)

        return rnn_embed_imu_hiddens, fusion_lstm_hiddens, fusion_features, out_features

    def init_data(self, poses, prev_belief):
        use_pose_model = True if type(poses) == PoseModel else False

        fusion_hiddens, fusion_features, out_features = [torch.empty(0)] * self.T, [torch.empty(0)] * self.T, [torch.empty(0)] * (self.T-1)
        fusion_hiddens[0], fusion_features[0] = prev_belief, prev_belief
        if self.args.belief_rnn == 'lstm':
            fusion_lstm_hiddens = fusion_hiddens = [torch.empty(0)] * self.T
            fusion_lstm_hiddens[0] = (prev_belief.unsqueeze(0).repeat(2, 1, 1), prev_belief.unsqueeze(0).repeat(2, 1, 1))
        
        if self.use_imu:
            running_batch_size = prev_belief.size()[0]
            rnn_embed_imu_hiddens = [(torch.empty(0))] * self.T
            prev_rnn_embed_imu_hidden = torch.zeros(2, running_batch_size, self.args.embedding_size, device=self.args.device)
            if self.args.imu_rnn == 'lstm':
                rnn_embed_imu_hiddens[0] = (prev_rnn_embed_imu_hidden, prev_rnn_embed_imu_hidden)
            elif self.args.imu_rnn == 'gru':
                rnn_embed_imu_hiddens[0] = prev_rnn_embed_imu_hidden

        return rnn_embed_imu_hiddens, fusion_lstm_hiddens, fusion_features, out_features

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
        use_pose_model = True if type(poses) == PoseModel else False
        T = self.args.clip_length + 1

        rnn_embed_imu_hiddens, \
            fusion_lstm_hiddens, \
            fusion_features, \
            out_features, \
            pred_poses = self.init_data(poses, prev_belief)
        
        for t in range(T - 1):
            t_ = t - 1 # Use t_ to deal with different time indexing for observations
            
            rnn_embed_imu_hiddens, \
                fusion_lstm_hiddens, \
                fusion_features, \
                out_features[t_ + 1], \
                pred_poses[t_ + 1] = self.execute_model(rnn_embed_imu_hiddens,
                                                        observations[1],
                                                        observations[0],
                                                        observations,
                                                        fusion_features,
                                                        fusion_lstm_hiddens,
                                                        poses,
                                                        pred_poses,
                                                        t)
        
        hidden = [torch.stack(fusion_features, dim=0), torch.stack(out_features, dim=0)]
        
        if use_pose_model:
            hidden += [torch.stack(pred_poses, dim=0)]
            if self.args.eval_uncertainty: hidden += [None]
        return hidden
