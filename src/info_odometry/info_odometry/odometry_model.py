import torch
import os

from info_odometry.info_model import bottle
from info_odometry.utils.tools import construct_models


class OdometryModel:
    def __init__(self, args):
        self.args = args
        self.model_dicts = None

        # use_imu: denote whether img and imu are used at the same time
        # args.imu_only: denote only imu is used
        (self.flownet_model,
         self.transition_model,
         self.use_imu,
         self.use_info,
         self.observation_model,
         self.observation_imu_model,
         self.pose_model,
         self.encoder) = construct_models(args)

        if args.finetune_only_decoder:
            assert args.finetune
            print("=> only finetune pose_model, while fixing encoder and transition_model")
        if args.finetune:
            assert args.load_model != "none"

        if args.finetune_only_decoder:
            self.param_list = list(self.pose_model.parameters())
        else:
            self.param_list = (list(self.transition_model.parameters()) +
                               list(self.pose_model.parameters()) +
                               list(self.encoder.parameters()))
            if self.observation_model:
                self.param_list += list(self.observation_model.parameters())
            if self.observation_imu_model:
                self.param_list += list(self.observation_imu_model.parameters())

    def load_model(self, model_path):
        # NOTE: Load prev ckp after we have specified optimizer
        assert os.path.exists(model_path)

        print("=> loading previous trained model: {}".format(model_path))
        self.model_dicts = torch.load(model_path, map_location=self.args.device)
        self.transition_model.load_state_dict(self.model_dicts["transition_model"], strict=True)
        if self.observation_model:
            self.observation_model.load_state_dict(self.model_dicts["observation_model"], strict=True)

        if self.observation_imu_model:
            self.observation_imu_model.load_state_dict(self.model_dicts["observation_imu_model"],
                                                       strict=True)
        self.pose_model.load_state_dict(self.model_dicts["pose_model"], strict=True)
        self.encoder.load_state_dict(self.model_dicts["encoder"], strict=True)

    def eval_flownet_model(self, x_img_pairs):
        # if we use flownet_feature as reconstructed observations -> no need for de-quantization
        with torch.no_grad():
            # [time, batch, out_conv6_1] e.g. [5, 16, 1024, 5, 19]
            observations = x_img_pairs if self.args.img_prefeat == 'flownet' else bottle(self.flownet_model,
                                                                                         (x_img_pairs,))
        return observations

    def run_train(self, x_img_pairs, x_imu_seqs, init_state, y_rel_poses, init_belief):
        self.pose_model.train()
        if self.args.finetune_only_decoder:
            self.encoder.eval()
            self.transition_model.eval()
            if self.observation_model:
                self.observation_model.eval()
            if self.observation_imu_model:
                self.observation_imu_model.eval()
        else:
            self.encoder.train()
            self.transition_model.train()
            if self.observation_model:
                self.observation_model.train()
            if self.observation_imu_model:
                self.observation_imu_model.train()

        observations = self.eval_flownet_model(x_img_pairs)

        obs_size = observations.size()
        observations = observations.view(obs_size[0], obs_size[1], -1)

        # update belief/state using posterior from previous belief/state, previous pose and current observation
        # (over entire sequence at once)
        # output: [time, ] with init states already removed
        if self.args.finetune_only_decoder:
            with torch.no_grad():
                if self.use_imu:
                    encode_observations = (bottle(self.encoder, (observations,)), x_imu_seqs)
                elif self.args.imu_only:
                    encode_observations = x_imu_seqs
                else:
                    encode_observations = bottle(self.encoder, (observations,))

                args_transition = {
                    'prev_state': init_state,  # not used if not use_info
                    'poses': y_rel_poses,  # not used if not use_info during training
                    'prev_belief': init_belief,
                    'observations': encode_observations
                }

                # if args.hard: args_transition['gumbel_temperature'] = gumbel_tau

                beliefs, \
                    prior_states, \
                    prior_means, \
                    prior_std_devs, \
                    posterior_states, \
                    posterior_means, \
                    posterior_std_devs = self.transition_model(**args_transition)

        else:
            if self.use_imu:
                encode_observations = (bottle(self.encoder, (observations,)), x_imu_seqs)
            elif self.args.imu_only:
                encode_observations = x_imu_seqs
            else:
                encode_observations = bottle(self.encoder, (observations,))

            args_transition = {
                'prev_state': init_state,  # not used if not use_info
                'poses': y_rel_poses,  # not used if not use_info during training
                'prev_belief': init_belief,
                'observations': encode_observations
            }

            # if args.hard: args_transition['gumbel_temperature'] = gumbel_tau

            beliefs, \
                prior_states, \
                prior_means, \
                prior_std_devs, \
                posterior_states, \
                posterior_means, \
                posterior_std_devs = self.transition_model(**args_transition)

        pred_rel_poses = bottle(self.pose_model, (posterior_states,))

        return beliefs, \
            prior_states, \
            prior_means, \
            prior_std_devs, \
            posterior_states, \
            posterior_means, \
            posterior_std_devs, \
            pred_rel_poses

    def run_eval(self, observations, x_imu_seqs, init_state, beliefs):
        self.pose_model.eval()
        self.encoder.eval()
        self.transition_model.eval()
        if self.observation_model:
            self.observation_model.eval()
        if self.observation_imu_model:
            self.observation_imu_model.eval()

        with torch.no_grad():
            obs_size = observations.size()
            observations = observations.view(obs_size[0], obs_size[1], -1)

            if self.use_imu:
                encode_observations = (bottle(self.encoder, (observations,)), x_imu_seqs)
            elif self.args.imu_only:
                encode_observations = x_imu_seqs
            else:
                encode_observations = bottle(self.encoder, (observations,))

            # with one more returns: poses
            beliefs, \
                prior_states, \
                prior_means, \
                prior_std_devs, \
                posterior_states, \
                posterior_means, \
                posterior_std_devs, \
                pred_rel_poses = self.transition_model(prev_state=init_state,  # not used if not use_info
                                                       poses=self.pose_model,
                                                       prev_belief=beliefs,
                                                       observations=encode_observations)

        return beliefs, \
            prior_states, \
            prior_means, \
            prior_std_devs, \
            posterior_states, \
            posterior_means, \
            posterior_std_devs, \
            pred_rel_poses
