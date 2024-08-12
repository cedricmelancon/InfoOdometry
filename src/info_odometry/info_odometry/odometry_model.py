import torch
import os

from info_odometry.model import (OdometryModelFlownet2S,
                    OdometryModelEncoder,
                    OdometryModelTransition,
                    OdometryModelPose)
from info_odometry.utils.tools import bottle


class OdometryModel:
    def __init__(self, args):
        self.index = None
        self.args = args
        self.model_dicts = None

        self.device = args.device

        self.flownet_model = None
        self.encoder = None
        self.transition_model = None
        self.pose_model = None

        if args.include_flownet:
            self.flownet_model = OdometryModelFlownet2S(args).to(device=self.device)
            self.flownet_model.load_model("")
            self.flownet_model.eval()

        base_args = {
            'args': args,
            'embedding_size': args.embedding_size,
            'activation_function': args.activation_function
        }

        if args.include_encoder:
            self.encoder = OdometryModelEncoder(**base_args).to(device=self.device)

        base_args = {
            'args': args,
            'embedding_size': args.embedding_size,
            'activation_function': args.activation_function,
            'belief_size': args.belief_size,
            'state_size': args.state_size,
            'hidden_size': args.hidden_size,
        }
        if args.include_transition:
            self.transition_model = OdometryModelTransition(**base_args).to(device=self.device)

        if args.include_pose:
            self.pose_model = OdometryModelPose(**base_args).to(device=self.device)

        self.clip_length = args.clip_length
        self.initialized = False

        self.beliefs = torch.rand(1, self.args.belief_size, device=self.args.device)

        if args.load_model != 'none':
            self.load_model(args.load_model)

        if args.finetune_only_decoder:
            self.param_list = list(self.pose_model.parameters())
        else:
            self.param_list = (list(self.transition_model.parameters()) +
                               list(self.pose_model.parameters()) +
                               list(self.encoder.parameters()))

    def load_model(self, model_path):
        # NOTE: Load prev ckp after we have specified optimizer
        assert os.path.exists(model_path)

        print("=> loading previous trained model: {}".format(model_path))
        self.model_dicts = torch.load(model_path, map_location=self.args.device)
        self.transition_model.load_state_dict(self.model_dicts["transition_model"], strict=True)
        self.pose_model.load_state_dict(self.model_dicts["pose_model"], strict=True)
        self.encoder.load_state_dict(self.model_dicts["encoder"], strict=True)

    def eval_flownet_model(self, x_img_pairs):
        # if we use flownet_feature as reconstructed observations -> no need for de-quantization
        with torch.no_grad():
            # [time, batch, out_conv6_1] e.g. [5, 16, 1024, 5, 19]
            observations = x_img_pairs if self.args.img_prefeat == 'flownet' else bottle(self.flownet_model,
                                                                                         (x_img_pairs,))
        return observations

    def run_train(self, x_img_pairs, x_imu_seqs, init_belief):
        self.pose_model.train()
        if self.args.finetune_only_decoder:
            self.encoder.eval()
            self.transition_model.eval()
        else:
            self.encoder.train()
            self.transition_model.train()

        observations = self.eval_flownet_model(x_img_pairs)

        obs_size = observations.size()
        observations = observations.view(obs_size[0], obs_size[1], -1)

        # update belief/state using posterior from previous belief/state, previous pose and current observation
        # (over entire sequence at once)
        # output: [time, ] with init states already removed
        if self.args.finetune_only_decoder:
            with torch.no_grad():
                encode_observations = (bottle(self.encoder, (observations,)), x_imu_seqs)
                beliefs, posterior_states = self.transition_model(init_belief, encode_observations)
        else:
            encode_observations = (bottle(self.encoder, (observations,)), x_imu_seqs)
            beliefs, posterior_states = self.transition_model(init_belief, encode_observations)

        pred_rel_poses = bottle(self.pose_model, (posterior_states,))

        return beliefs, pred_rel_poses

    def run_eval(self, observations, x_imu_seqs, beliefs):
        self.pose_model.eval()
        self.encoder.eval()
        self.transition_model.eval()

        with torch.no_grad():
            obs_size = observations.size()
            observations = observations.view(obs_size[0], obs_size[1], -1)

            encode_observations = (bottle(self.encoder, (observations,)), x_imu_seqs)

            beliefs, posterior_states = self.transition_model(beliefs, encode_observations)
            pred_rel_poses = bottle(self.pose_model, (posterior_states,))

        return beliefs, pred_rel_poses

    def set_eval(self):
        self.pose_model.eval()
        self.encoder.eval()
        self.transition_model.eval()

    def step_model(self, observations, x_imu_seqs, init_state):
        if not self.initialized:
            self.set_eval()

            prev_beliefs = torch.rand(1, self.args.belief_size, device=self.args.device)
            (self.rnn_embed_imu_hiddens, self.fusion_lstm_hiddens,
             self.fusion_features, self.out_features) = self.transition_model.init_data(prev_beliefs)

            self.index = 0
            self.initialized = True
        
        pred_rel_poses = None

        with torch.no_grad():
            obs_size = observations.size()
            observations = observations.view(obs_size[0], obs_size[1], -1)
            
            encode_observations = (self.encoder(observations), x_imu_seqs)

            (self.rnn_embed_imu_hiddens,
             self.fusion_lstm_hiddens,
             self.fusion_features,
             self.out_features[self.index]) = self.transition_model.execute_model(self.rnn_embed_imu_hiddens,
                                                                                  encode_observations[1],
                                                                                  encode_observations[0],
                                                                                  self.fusion_features,
                                                                                  self.fusion_lstm_hiddens,
                                                                                  self.index)

            if self.index == self.clip_length - 1:
                pred_rel_poses = self.pose_model(self.out_features[self.index])

        if self.index < self.clip_length - 1:
            self.index += 1

        return pred_rel_poses
