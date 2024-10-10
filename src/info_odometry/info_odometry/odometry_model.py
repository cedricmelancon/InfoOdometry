import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

import time

from info_odometry.modules.observation_model import ObservationModel
from info_odometry.modules.seq_vi_net import SeqVINet
from info_odometry.modules.single_hidden_transition_model import SingleHiddenTransitionModel
from info_odometry.modules.double_hidden_transition_model import DoubleHiddenTransitionModel
from info_odometry.modules.single_hidden_vi_transition_model import SingleHiddenVITransitionModel
from info_odometry.modules.double_hidden_vi_transition_model import DoubleHiddenVITransitionModel
from info_odometry.modules.multi_hidden_vi_transition_model import MultiHiddenVITransitionModel
from info_odometry.modules.double_stochastic_transition_model import DoubleStochasticTransitionModel
from info_odometry.modules.double_stochastic_vi_transition_model import DoubleStochasticVITransitionModel
from info_odometry.modules.pose_model import PoseModel
from info_odometry.modules.encoder import Encoder
from info_odometry.flownet_model import FlowNet2S

from info_odometry.utils.model_utils import ModelUtils


class OdometryModel:
    def __init__(self, args):
        self.step_tau = None
        self.gumbel_tau = None
        self.anneal_epochs = None
        self._final_tau = None
        self._start_tau = None
        self._optimizer = None
        self.flownet_model = None
        self.observation_model = None
        self.observation_imu_model = None
        self.encoder = None
        self.pose_model = None
        self.args = args
        self.param_list = None

        self.use_info = None

        self.clip_length = args.clip_length
        self.initialized = False

        self._construct_models()

        if self.use_info:
            # global prior N(0, I)
            self._global_prior = Normal(
                torch.zeros(self.args.batch_size, self.args.state_size, device=self.args.device),
                torch.ones(self.args.batch_size, self.args.state_size, device=self.args.device))

            # allowed deviation in kl divergence
            self._free_nats = torch.full((1,), self.args.free_nats, device=self.args.device)

        # gumbel temperature for hard deepvio
        if self.args.hard:
            self._start_tau = self.args.gumbel_tau_start  # default 1.0
            self._final_tau = self.args.gumbel_tau_final  # default 0.5
            self.anneal_epochs = int(self.args.epoch * self.args.gumbel_tau_epoch_ratio)
            self.step_tau = (self._start_tau - self._final_tau) / (self.anneal_epochs - 1)
            self.gumbel_tau = self._start_tau + self.step_tau

    def init_eval(self):
        self.eval()


    def train(self):
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

    def eval(self):
        self.pose_model.eval()
        self.encoder.eval()
        self.transition_model.eval()
        if self.observation_model:
            self.observation_model.eval()
        if self.observation_imu_model:
            self.observation_imu_model.eval()

    def forward_flownet(self, x_img_pairs):
        # [time, batch, 3, 2, H, W]
        #x_img_pairs = torch.stack(x_img_list, dim=0).type(torch.FloatTensor).to(device=self.args.device)
        #self._running_batch_size = x_img_pairs.size()[1]  # might be different for the last batch

        # if we use flownet_feature as reconstructed observations -> no need for dequantization
        with torch.no_grad():
            # [time, batch, out_conv6_1] e.g. [5, 16, 1024, 5, 19]
            if self.args.img_prefeat == 'flownet':
                observations = x_img_pairs
            else:
                observations = ModelUtils.bottle(self.flownet_model, (x_img_pairs,))

        obs_size = observations.size()
        observations = observations.view(obs_size[0], obs_size[1], -1)

        return observations

    def forward_full(self, x_img_pairs, x_imu_seqs, prev_beliefs):
        start_time = time.perf_counter()
        # if self._use_imu or self.args.imu_only:
        # [time, batch, 11, 6]
        # x_imu_seqs = torch.stack(x_imu_list, dim=0).type(torch.FloatTensor).to(device=self.args.device)
        observations = ModelUtils.bottle(self.flownet_model, (x_img_pairs,))
        obs_size = observations.size()
        observations = observations.view(obs_size[0], obs_size[1], -1)

        # update belief/state using posterior from previous belief/state, previous pose and current
        # observation (over entire sequence at once)
        # output: [time, ] with init states already removed
        # if self.args.finetune_only_decoder:
        #     with (torch.no_grad()):
        #         if self._use_imu:
        #             encode_observations = (ModelUtils.bottle(self.encoder, (observations,)), x_imu_seqs)
        #         elif self.args.imu_only:
        #             encode_observations = x_imu_seqs
        #         else:
        #             encode_observations = ModelUtils.bottle(self.encoder, (observations,))
        #
        #         args_transition = {
        #             'prev_state': prev_state,  # not used if not use_info
        #             'poses': y_rel_poses,  # not used if not use_info during training
        #             'prev_belief': prev_beliefs,
        #             'observations': encode_observations
        #         }
        #
        #         if self.args.hard:
        #             args_transition['gumbel_temperature'] = self.gumbel_tau
        #
        #         (beliefs,
        #          prior_states,
        #          prior_means,
        #          prior_std_devs,
        #          posterior_states,
        #          posterior_means,
        #          posterior_std_devs) = self.transition_model(**args_transition)
        # else:
        if self._use_imu:
            encode_observations = (ModelUtils.bottle(self.encoder, (observations,)), x_imu_seqs)
        elif self.args.imu_only:
            encode_observations = x_imu_seqs
        else:
            encode_observations = ModelUtils.bottle(self.encoder, (observations,))

        encoder_time = time.perf_counter()

        args_transition = {
            'prev_state': None,  # prev_state,  # not used if not use_info
            'poses': None,  # y_rel_poses,  # not used if not use_info during training
            'prev_belief': prev_beliefs,
            'observations': encode_observations
        }

        if self.args.hard:
            args_transition['gumbel_temperature'] = self.gumbel_tau

        (beliefs,
         prior_states,
         prior_means,
         prior_std_devs,
         posterior_states,
         posterior_means,
         posterior_std_devs) = self.transition_model(**args_transition)

        transition_time = time.perf_counter()
        # (pred_observations,
        #  pred_imu_observations) = self._forward_observation(beliefs,
        #                                                     posterior_states,
        #                                                     prior_states,
        #                                                     prior_means,
        #                                                     prior_std_devs,
        #                                                     observations,
        #                                                     x_imu_seqs,
        #                                                     posterior_means,
        #                                                     posterior_std_devs)

        pred_rel_poses = ModelUtils.bottle(self.pose_model, (posterior_states,))
        pose_time = time.perf_counter()
        timing = [encoder_time - start_time, transition_time - encoder_time, pose_time - transition_time]
        return (beliefs,
                pred_rel_poses,
                timing)

    def forward(self, observations, x_imu_seqs, prev_beliefs):
        start_time = time.perf_counter()
        #if self._use_imu or self.args.imu_only:
            # [time, batch, 11, 6]
            #x_imu_seqs = torch.stack(x_imu_list, dim=0).type(torch.FloatTensor).to(device=self.args.device)

        # update belief/state using posterior from previous belief/state, previous pose and current
        # observation (over entire sequence at once)
        # output: [time, ] with init states already removed
        # if self.args.finetune_only_decoder:
        #     with (torch.no_grad()):
        #         if self._use_imu:
        #             encode_observations = (ModelUtils.bottle(self.encoder, (observations,)), x_imu_seqs)
        #         elif self.args.imu_only:
        #             encode_observations = x_imu_seqs
        #         else:
        #             encode_observations = ModelUtils.bottle(self.encoder, (observations,))
        #
        #         args_transition = {
        #             'prev_state': prev_state,  # not used if not use_info
        #             'poses': y_rel_poses,  # not used if not use_info during training
        #             'prev_belief': prev_beliefs,
        #             'observations': encode_observations
        #         }
        #
        #         if self.args.hard:
        #             args_transition['gumbel_temperature'] = self.gumbel_tau
        #
        #         (beliefs,
        #          prior_states,
        #          prior_means,
        #          prior_std_devs,
        #          posterior_states,
        #          posterior_means,
        #          posterior_std_devs) = self.transition_model(**args_transition)
        # else:
        if self._use_imu:
            encode_observations = (ModelUtils.bottle(self.encoder, (observations,)), x_imu_seqs)
        elif self.args.imu_only:
            encode_observations = x_imu_seqs
        else:
            encode_observations = ModelUtils.bottle(self.encoder, (observations,))

        encoder_time = time.perf_counter()

        args_transition = {
            'prev_state': None,  # prev_state,  # not used if not use_info
            'poses': None,  # y_rel_poses,  # not used if not use_info during training
            'prev_belief': prev_beliefs,
            'observations': encode_observations
        }

        if self.args.hard:
            args_transition['gumbel_temperature'] = self.gumbel_tau

        (beliefs,
         prior_states,
         prior_means,
         prior_std_devs,
         posterior_states,
         posterior_means,
         posterior_std_devs) = self.transition_model(**args_transition)

        transition_time = time.perf_counter()
        # (pred_observations,
        #  pred_imu_observations) = self._forward_observation(beliefs,
        #                                                     posterior_states,
        #                                                     prior_states,
        #                                                     prior_means,
        #                                                     prior_std_devs,
        #                                                     observations,
        #                                                     x_imu_seqs,
        #                                                     posterior_means,
        #                                                     posterior_std_devs)

        pred_rel_poses = ModelUtils.bottle(self.pose_model, (posterior_states,))
        pose_time = time.perf_counter()
        timing = [encoder_time - start_time, transition_time - encoder_time, pose_time - transition_time]
        return (beliefs,
                pred_rel_poses,
                timing)

    # def _compute_observation_loss(self, pred_observations, observations, pred_imu_observations, x_imu_seqs):
    #     observation_loss = None
    #     observation_imu_loss = None
    #     kl_loss = None
    #
    #     if self.args.rec_loss == 'sum':
    #         # might be too large -> if so: .mean(dim=2).mean(dim=(0,1)) instead
    #         observation_loss = (F.mse_loss(pred_observations,
    #                                        observations,
    #                                        reduction='none').sum(dim=2).mean(dim=(0, 1)))
    #         # observation_loss = F.l1_loss(pred_observations, observations, reduction='none').sum(
    #         #    dim=2).mean(dim=(0, 1))  # might be too large -> if so: .mean(dim=2).mean(dim=(0,1)) instead
    #     elif self.args.rec_loss == 'mean':
    #         # might be too large -> if so: .mean(dim=2).mean(dim=(0,1)) instead
    #         observation_loss = F.mse_loss(pred_observations,
    #                                       observations,
    #                                       reduction='none').mean(dim=2).mean(dim=(0, 1))
    #         # observation_loss = F.l1_loss(pred_observations, observations, reduction='none').mean(
    #         #    dim=2).mean(dim=(0, 1))  # might be too large -> if so: .mean(dim=2).mean(dim=(0,1)) instead
    #     observation_loss = self.args.observation_beta * observation_loss
    #
    #     observation_imu_loss = self._compute_observation_imu_loss(pred_imu_observations, x_imu_seqs)
    #
    #     if self.args.kl_free_nats == 'none':
    #         kl_loss = (self.args.world_kl_beta *
    #                    kl_divergence(Normal(posterior_means, posterior_std_devs),
    #                                  Normal(prior_means, prior_std_devs)).sum(dim=2).mean(dim=(0, 1)))
    #     elif self.args.kl_free_nats == 'min':
    #         if self.args.world_kl_out:
    #             kl_loss = (self.args.world_kl_beta *
    #                        torch.min(kl_divergence(Normal(posterior_means, posterior_std_devs),
    #                                                Normal(prior_means, prior_std_devs)).sum(dim=2),
    #                                  self._free_nats).mean(dim=(0, 1)))
    #         else:
    #             kl_loss = torch.min(self.args.world_kl_beta *
    #                                 kl_divergence(Normal(posterior_means, posterior_std_devs),
    #                                               Normal(prior_means, prior_std_devs)).sum(dim=2),
    #                                 self._free_nats).mean(dim=(0, 1))
    #     elif self.args.kl_free_nats == 'max':
    #         if self.args.world_kl_out:
    #             kl_loss = (self.args.world_kl_beta *
    #                        torch.max(kl_divergence(Normal(posterior_means, posterior_std_devs),
    #                                                Normal(prior_means, prior_std_devs)).sum(dim=2),
    #                                  self._free_nats).mean(dim=(0, 1)))
    #         else:
    #             kl_loss = torch.max(self.args.world_kl_beta *
    #                                 kl_divergence(Normal(posterior_means, posterior_std_devs),
    #                                               Normal(prior_means, prior_std_devs)).sum(dim=2),
    #                                 self._free_nats).mean(dim=(0, 1))
    #     if self.args.global_kl_beta != 0:
    #         if self._running_batch_size == self.args.batch_size:
    #             kl_loss += (self.args.global_kl_beta *
    #                         kl_divergence(Normal(posterior_means, posterior_std_devs),
    #                                       self._global_prior).sum(dim=2).mean(dim=(0, 1)))
    #         else:
    #             tmp_global_prior = Normal(torch.zeros(running_batch_size, args.state_size, device=args.device),
    #                                       torch.ones(running_batch_size, args.state_size, device=args.device))
    #             kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs),
    #                                                            tmp_global_prior).sum(dim=2).mean(dim=(0, 1))
    #
    #     return observation_loss, observation_imu_loss, kl_loss
    #
    # def _compute_observation_imu_loss(self, pred_imu_observations, x_imu_seqs):
    #     observation_imu_loss = None
    #
    #     if self.args.rec_loss == 'sum':
    #         # might be too large -> if so: .mean(dim=2).mean(dim=(0,1)) instead
    #         observation_imu_loss = F.mse_loss(pred_imu_observations,
    #                                           x_imu_seqs.view(pred_imu_observations.size()),
    #                                           reduction='none').sum(dim=2).mean(dim=(0, 1))
    #         # observation_imu_loss = F.l1_loss(pred_imu_observations,
    #         #                                  x_imu_seqs.view(pred_imu_observations.size()),
    #         #                                  reduction='none').sum(dim=2).mean(
    #         #    dim=(0, 1))  # might be too large -> if so: .mean(dim=2).mean(dim=(0,1)) instead
    #     elif self.args.rec_loss == 'mean':
    #         # might be too large -> if so: .mean(dim=2).mean(dim=(0,1)) instead
    #         observation_imu_loss = F.mse_loss(pred_imu_observations,
    #                                           x_imu_seqs.view(pred_imu_observations.size()),
    #                                           reduction='none').mean(dim=2).mean(dim=(0, 1))
    #         # observation_imu_loss = F.l1_loss(pred_imu_observations,
    #         #                                  x_imu_seqs.view(pred_imu_observations.size()),
    #         #                                  reduction='none').mean(dim=2).mean(
    #         #    dim=(0, 1))  # might be too large -> if so: .mean(dim=2).mean(dim=(0,1)) instead
    #     observation_imu_loss = self.args.observation_imu_beta * observation_imu_loss
    #
    #     return observation_imu_loss

    # def _forward_observation(self,
    #                          beliefs,
    #                          posterior_states,
    #                          prior_states,
    #                          prior_means,
    #                          prior_std_devs,
    #                          observations,
    #                          x_imu_seqs,
    #                          posterior_means,
    #                          posterior_std_devs):
    #     pred_observations = None
    #     pred_imu_observations = None
    #
    #     if self._use_info:
    #         # observation reconstruction for images
    #         if self.args.observation_beta != 0:
    #             if self._use_imu:
    #                 beliefs_visual = beliefs[0]
    #             else:
    #                 beliefs_visual = beliefs
    #
    #             if self.args.finetune_only_decoder:
    #                 with torch.no_grad():
    #                     if self.args.rec_type == 'posterior':
    #                         pred_observations = ModelUtils.bottle(self.observation_model,
    #                                                               (beliefs_visual, posterior_states,))
    #                     elif self.args.rec_type == 'prior':
    #                         pred_observations = ModelUtils.bottle(self.observation_model,
    #                                                               (beliefs_visual, prior_states,))
    #             else:
    #                 if self.args.rec_type == 'posterior':
    #                     pred_observations = ModelUtils.bottle(self.observation_model,
    #                                                           (beliefs_visual, posterior_states,))
    #                 elif self.args.rec_type == 'prior':
    #                     pred_observations = ModelUtils.bottle(self.observation_model,
    #                                                           (beliefs_visual, prior_states,))
    #
    #         # observation reconstruction for imus
    #         if self._use_imu and self.args.observation_imu_beta != 0:
    #             if self.args.finetune_only_decoder:
    #                 with torch.no_grad():
    #                     if self.args.rec_type == 'posterior':
    #                         pred_imu_observations = ModelUtils.bottle(self.observation_imu_model,
    #                                                                   (beliefs[1], posterior_states,))
    #                     elif self.args.rec_type == 'prior':
    #                         pred_imu_observations = ModelUtils.bottle(self.observation_imu_model,
    #                                                                   (beliefs[1], prior_states,))
    #             else:
    #                 if self.args.rec_type == 'posterior':
    #                     pred_imu_observations = ModelUtils.bottle(self.observation_imu_model,
    #                                                               (beliefs[1], posterior_states,))
    #                 elif self.args.rec_type == 'prior':
    #                     pred_imu_observations = ModelUtils.bottle(self.observation_imu_model,
    #                                                               (beliefs[1], prior_states,))
    #
    #     return pred_observations, pred_imu_observations

    def _construct_flownet_model(self):
        if self.args.flownet_model != 'none' and self.args.img_prefeat == 'none':
            if self.args.train_img_from_scratch:
                raise ValueError('if --flownet_model -> --train_img_from_scratch should not be used')

            if self.args.flownet_model == 'FlowNet2S':
                self.flownet_model = FlowNet2S(self.args).to(device=self.args.device)
            else:
                raise ValueError('--flownet_model: {} is not supported'.format(self.args.flownet_model))
            resume_ckp = '/data/results/ckp/pretrained_flownet/{}_checkpoint.pth.tar'.format(
                self.args.flownet_model)
            flow_ckp = torch.load(resume_ckp)
            self.flownet_model.load_state_dict(flow_ckp['state_dict'])
            self.flownet_model.eval()

    def _construct_transition_model(self, base_args):
        self._use_imu = False
        self._use_info = False
        if self.args.transition_model == 'deepvo':
            self.transition_model = SeqVINet(use_imu=False, **base_args).to(device=self.args.device)
        elif self.args.transition_model == 'deepvio':
            self.transition_model = SeqVINet(use_imu=True, **base_args).to(device=self.args.device)
            self._use_imu = True
        elif self.args.transition_model == 'single':
            self.transition_model = SingleHiddenTransitionModel(**base_args).to(device=self.args.device)
            self._use_info = True
        elif self.args.transition_model == 'double':
            self.transition_model = DoubleHiddenTransitionModel(**base_args).to(device=self.args.device)
            self._use_info = True
        elif self.args.transition_model == 'double-stochastic':
            self.transition_model = DoubleStochasticTransitionModel(**base_args).to(device=self.args.device)
            self._use_info = True
        elif self.args.transition_model == 'single-vinet':
            self.transition_model = SingleHiddenVITransitionModel(**base_args).to(device=self.args.device)
            self._use_imu = True
            self._use_info = True
        elif self.args.transition_model == 'double-vinet':
            self.transition_model = DoubleHiddenVITransitionModel(**base_args).to(device=self.args.device)
            self._use_imu = True
            self._use_info = True
        elif self.args.transition_model == 'double-vinet-stochastic':
            self.transition_model = DoubleStochasticVITransitionModel(**base_args).to(device=self.args.device)
            self._use_imu = True
            self._use_info = True
        elif self.args.transition_model == 'multi-vinet':
            self.transition_model = MultiHiddenVITransitionModel(**base_args).to(device=self.args.device)
            self._use_imu = True
            self._use_info = True

        if self.args.transition_model not in ['deepvo', 'deepvio']:
            if self.args.observation_beta != 0:
                self.observation_model = ObservationModel(symbolic=True,
                                                          observation_type='visual',
                                                          **base_args).to(device=self.args.device)
            if self._use_imu and self.args.observation_imu_beta != 0:
                self.observation_imu_model = ObservationModel(symbolic=True,
                                                              observation_type='imu',
                                                              **base_args).to(device=self.args.device)

    def _construct_models(self):
        # check the validity of soft / hard deepvio
        if self.args.soft or self.args.hard:
            if self.args.transition_model not in ['deepvio', 'double-vinet']:
                raise ValueError('--soft and --hard must be used with deepvio or double-vinet')

        base_args = {
            'args': self.args,
            'belief_size': self.args.belief_size,
            'state_size': self.args.state_size,
            'hidden_size': self.args.hidden_size,
            'embedding_size': self.args.embedding_size,
            'activation_function': self.args.activation_function
        }

        # construct flownet_model
        self._construct_flownet_model()

        # construct transition model
        self._construct_transition_model(base_args)

        self.pose_model = PoseModel(**base_args).to(device=self.args.device)
        self.encoder = Encoder(symbolic=True, **base_args).to(device=self.args.device)

        if self.args.finetune_only_decoder:
            assert self.args.finetune
            print("=> only finetune pose_model, while fixing encoder and transition_model")
        if self.args.finetune:
            assert self.args.load_model != "none"

        if self.args.finetune_only_decoder:
            self.param_list = list(self.pose_model.parameters())
        else:
            self.param_list = (list(self.transition_model.parameters()) +
                                list(self.pose_model.parameters()) +
                                list(self.encoder.parameters()))
            if self.observation_model:
                self.param_list += list(self.observation_model.parameters())
            if self.observation_imu_model:
                self.param_list += list(self.observation_imu_model.parameters())

    def load_model(self):
        print("=> loading previous trained model: {}".format(self.args.load_model))
        model_dicts = torch.load(self.args.load_model, map_location="cuda:0")
        self.transition_model.load_state_dict(model_dicts["transition_model"], strict=True)
        if self.observation_model:
            self.observation_model.load_state_dict(model_dicts["observation_model"], strict=True)
        if self.observation_imu_model:
            self.observation_imu_model.load_state_dict(model_dicts["observation_imu_model"], strict=True)
        self.pose_model.load_state_dict(model_dicts["pose_model"], strict=True)
        self.encoder.load_state_dict(model_dicts["encoder"], strict=True)

        optimizer_dict = None
        #if not self.args.finetune:
            #optimizer_dict = model_dicts["optimizer"]

        return optimizer_dict

