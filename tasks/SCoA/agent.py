''' Agents: stop/random/shortest/seq2seq  '''

import json
import os
import sys
import numpy as np
import random
import time
import math

import torch
import torch.nn as nn
import torch.distributions as D
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import utils
from env import R2RBatch
from utils import padding_idx
from param import args
from collections import defaultdict

import scipy.stats

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(args.seed)
        self.results = {} 
        self.losses = [] # For learning agents
        self.IL_losses = []
        self.WeTA_losses = []
        self.RL_losses = []
        self.WaTA_losses = []
    
    def write_results(self):
        output = [{'inst_idx': k, 'trajectory': v} for k, v in self.results.iteritems()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def rollout(self):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]
    
    def test(self):
        self.env.reset_epoch()
        self.losses = []
        self.WeTA_losses = []
        self.RL_losses = []
        self.WaTA_losses = []
        self.IL_losses = []

        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        #print 'Testing %s' % self.__class__.__name__
        looped = False
        while True:
            for traj in self.rollout():
                if traj['inst_idx'] in self.results:
                    looped = True
                else:
                    self.results[traj['inst_idx']] = traj['path']
            if looped:
                break
   
    
class AskOracle(object):
    def __init__(self):
        self.uncertain_threshold = args.uncertain_threshold
        self.DONT_ASK = 0
        self.ASK = 1

    def isUncertain(self, agent_dist):
        uniform = [1. / len(agent_dist)] * len(agent_dist)
        entropy_gap = scipy.stats.entropy(uniform) - scipy.stats.entropy(agent_dist)
        if entropy_gap < self.uncertain_threshold - 1e-9:
            return self.ASK, 'uncertain '
        else:
            return self.DONT_ASK, ''

    def __call__(self, agent_dist, last_distance, cur_distance):
        finalreason = ""
        ask_1, ask_reason_1 = self.isUncertain(agent_dist)
        if ask_1 == 1:
            finalreason += ask_reason_1

        if last_distance - cur_distance <= 0: # no progress
            ask_2 = 1
            ask_reason_2 = 'no progress'
        else:
            ask_2 = 0
            ask_reason_2 = ''

        if ask_2 == 1:
            finalreason += ask_reason_2

        finalask = 1 if (ask_1 or ask_2) else 0

        return finalask, finalreason


class SCoA_Agent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = {
      "left": ([0],[-1], [0]), # left
      "right": ([0], [1], [0]), # right
      "up": ([0], [0], [1]), # up
      "down": ([0], [0],[-1]), # down
      "forward": ([1], [0], [0]), # forward
      "<end>": ([0], [0], [0]), # <end>
      "<start>": ([0], [0], [0]), # <start>
      "<ignore>": ([0], [0], [0])  # <ignore>
    }
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, critic, WeTA, episode_len=20, visable=False, path_type='planner_path'):
        super(SCoA_Agent, self).__init__(env, results_path)
        self.feature_size = 2048
        self.encoder = encoder
        self.decoder = decoder

        self.critic = critic
        self.WeTA = WeTA

        self.episode_len = episode_len
        self.losses = []
        self.IL_losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)

        self.WeTA_criterion = nn.CrossEntropyLoss(size_average=False)
        self.WeTA_losses = []

        self.RL_losses = []
        self.WaTA_losses = []

        self.logs = defaultdict(list)


    @staticmethod
    def n_inputs():
        return len(SCoA_Agent.model_actions)

    @staticmethod
    def n_outputs():
        return len(SCoA_Agent.model_actions)-2 # Model doesn't output start or ignore

    # prepare candidate length and feature(bs, max_candidate_leng, 2048+4)
    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = c['feature']                         # Image feat
        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    # get angle feature, image feature, candidate feature, candidate leng
    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()

        f_t = self._feature_variable(obs)      # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        # input_a_t: shape(bs, angle_feat_size)
        # f_t: shape(bs, 36, 2048)
        # candidate_feat: shape(bs, max_candidate_leng, 2048+4)
        # candidate_leng: shape(bs, 1)
        return input_a_t, f_t, candidate_feat, candidate_leng

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.num_view, self.feature_size + args.angle_feat_size), dtype=np.float32)
        # feature_size = obs[0]['feature'].shape[0]
        # features = np.empty((len(obs), self.feature_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature']
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def get_nav_feature_variable(self, obs):
        features = np.empty((len(obs), 5, args.num_view, self.feature_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :] = ob['nav_history_feature']
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    try:
                        assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    except:
                        import pdb; pdb.set_trace()
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()[0]
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def update_ctx(self, ctx, h_t, c_t, ans_ctx, ans_h_t, ans_c_t, predict_WeTA):
        predict_WeTA = predict_WeTA.unsqueeze(-1) # (bs, 1)

        ctx = ctx + predict_WeTA * ans_ctx
        h_t = h_t + predict_WeTA * ans_h_t
        c_t = c_t + predict_WeTA * ans_c_t

        return ctx, h_t, c_t

    def getangle_feature(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i,ob in enumerate(obs):
            features[i,:] = ob['feature'][ob['viewIndex']]
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def rollout(self, iter=None, train_RL=True):

        obs = np.array(self.env.reset())
        batch_size = len(obs)

        # Record starting point
        traj = [{
            'inst_idx': ob['inst_idx'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']

        # ctx: shape(bs, 1, 512)
        # h_t: shape(bs, 512)
        # c_t: shape(bs, 512)
        # tar_enc: shape(bs, 1, 512)
        ctx, h_t, c_t, tar_enc = self.encoder(obs, cur_img=None, next_img=None, tar_enc=None, encoder_target=True)

        # Initialization the tracking state
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # For test result submission
        visited = [set() for _ in obs]

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        self.IL_loss = 0
        self.WeTA_loss = 0
        self.RL_loss = 0
        self.WaTA_loss = 0
        
        h1 = h_t
        for t in range(self.episode_len):
            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(obs)

            WeTA_logit, predict_WeTA = self.WeTA(h_t)  # (bs, )

            cur_imgfeature = self._feature_variable(obs)

            next_vp_imgfeature = np.array([ob['next_vp_imgfeature'] for ob in obs])
            next_vp_imgfeature = Variable(torch.from_numpy(next_vp_imgfeature), requires_grad=False).cuda()

            WaTA_loss, ans_ctx, ans_h_t, ans_c_t = \
                    self.encoder(obs=obs, cur_img=cur_imgfeature, next_img=next_vp_imgfeature, tar_enc=tar_enc)
            # udpate ctx, h_t, c_t
            ctx, h_t, c_t = self.update_ctx(ctx, h_t, c_t, ans_ctx, ans_h_t, ans_c_t, predict_WeTA)

            # add env dropout
            noise = self.decoder.drop_env(torch.ones(self.feature_size).cuda())
            candidate_feat[..., :-args.angle_feat_size] *= noise
            f_t[..., :-args.angle_feat_size] *= noise

            h_t, c_t, logit, h1 = self.decoder(input_a_t, f_t, candidate_feat,
                                    h_t, h1, c_t, ctx, None)

            hidden_states.append(h_t)
            candidate_mask = utils.length2mask(candidate_leng)

            # if 'test' in self.env.splits:
            if args.submit == 'True':     # Avoding cyclic path
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            candidate_mask[ob_id][c_id] = 1
            logit.masked_fill_(candidate_mask, -float('inf'))
            
            # Supervised training
            if 'test' not in self.env.splits:
                self.loss += WaTA_loss
                self.WaTA_loss += WaTA_loss

                # WeTA
                WeTA_target = []
                for ob_id, ob in enumerate(obs):
                    if ended[ob_id]:
                        WeTA_target.append(AskOracle().DONT_ASK)
                    else:
                        WeTA_t, _ = AskOracle()(logit[ob_id].cpu().detach().numpy(), last_dist[ob_id], ob['distance'])
                        WeTA_target.append(WeTA_t)
                WeTA_target = Variable(torch.Tensor(WeTA_target).long(), requires_grad=False).cuda()

                WeTA_loss = self.WeTA_criterion(WeTA_logit, WeTA_target)
                self.loss += WeTA_loss
                self.WeTA_loss += WeTA_loss

                target = self._teacher_action(obs, ended)

                IL_loss = self.criterion(logit, target)
                if not math.isinf(IL_loss):
                    self.loss += IL_loss
                    self.IL_loss += IL_loss

            probs = F.softmax(logit, dim=1)
            c = D.Categorical(probs)
            # RL
            self.logs['entropy'].append(c.entropy().sum().item())  # For log
            entropys.append(c.entropy())  # For optimization
            a_t = c.sample().detach()
            policy_log_probs.append(c.log_prob(a_t))

            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid:    # The last action is <end>
                    cpu_a_t[i] = -1

            self.make_equiv_action(cpu_a_t, obs, None, traj)
            obs = np.array(self.env._get_obs())

            dist = np.zeros(batch_size, np.float32)
            reward = np.zeros(batch_size, np.float32)
            mask = np.ones(batch_size, np.float32)
            for i, ob in enumerate(obs):
                dist[i] = ob['distance']
                if ended[i]:            # If the action is already finished BEFORE THIS ACTION.
                    reward[i] = 0.
                    mask[i] = 0.
                else:       # Calculate the reward            
                    if predict_WeTA[i] >= 0.5:
                        reward[i] = args.ask_reward
                    else:
                        reward[i] = 0
                    action_idx = cpu_a_t[i]
                    if action_idx == -1:        # If the action now is end
                        if dist[i] < 3:         # Correct
                            reward[i] = reward[i] + 3.
                        else:                   # Incorrect
                            reward[i] = reward[i] - 3.
                    else:                       # The action is not end
                        pro = - (dist[i] - last_dist[i])      # Change of distance
                        if pro > 0:                           # Quantification
                            reward[i] = reward[i] + 2
                        elif pro <= 0:
                            reward[i] = reward[i] - 2
                    
            rewards.append(reward)
            masks.append(mask)
            last_dist[:] = dist

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all(): 
                break

            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        # Last action in A2C
        input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(obs)
        candidate_feat[..., :-args.angle_feat_size] *= noise
        f_t[..., :-args.angle_feat_size] *= noise

        last_h_, _, _, _ = self.decoder(input_a_t, f_t, candidate_feat,
                         h_t, h1, c_t, ctx, None)
        RL_loss = 0.

        # NOW, A2C!!!
        # Calculate the final discounted reward
        last_value__ = self.critic(last_h_).detach()    # The value esti of the last state, remove the grad for safety
        discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
        for i in range(batch_size):
            if not ended[i]:        # If the action is not ended, use the value function as the last reward
                discount_reward[i] = last_value__[i]

        length = len(rewards)
        total = 0
        for t in range(length-1, -1, -1):
            discount_reward = discount_reward * args.gamma + rewards[t]   # If it ended, the reward will be 0
            mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
            clip_reward = discount_reward.copy()
            r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
            v_ = self.critic(hidden_states[t])
            a_ = (r_ - v_).detach()

            # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
            RL_loss += (-policy_log_probs[t] * a_ * mask_).sum()
            RL_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5     # 1/2 L2 loss
            if self.feedback == 'sample':
                RL_loss += (- 0.01 * entropys[t] * mask_).sum()
            self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

            total = total + np.sum(masks[t])
        self.logs['total'].append(total)


        self.RL_loss += RL_loss
        self.loss += RL_loss

        if 'test' not in self.env.splits:
            self.losses.append(self.loss.item() / self.episode_len)
            self.IL_losses.append(self.IL_loss.item() / self.episode_len)
            self.WeTA_losses.append(self.WeTA_loss.item() / self.episode_len)
            self.WaTA_losses.append(self.WaTA_loss.item() / self.episode_len)
            self.RL_losses.append(self.RL_loss.item() / self.episode_len)

        torch.cuda.empty_cache()

        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
            self.critic.train()
            self.WeTA.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.critic.eval()
            self.WeTA.eval()
        with torch.no_grad():
            super(SCoA_Agent, self).test()

    def train(self, encoder_optimizer, decoder_optimizer, critic_optimizer, WeTA_optimizer,
              n_iters, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.critic.train()
        self.WeTA.train()

        self.losses = []
        self.WeTA_losses = []
        self.RL_losses = []
        self.WaTA_losses = []
        self.IL_losses = []
        for iter in range(1, n_iters + 1):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            WeTA_optimizer.zero_grad()

            self.rollout(iter)
            self.loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            critic_optimizer.step()
            WeTA_optimizer.step()

    def save(self, encoder_path, decoder_path, critic_path, WeTA_path):
        ''' Snapshot models '''
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.WeTA.state_dict(), WeTA_path)

    def load(self, encoder_path, decoder_path, critic_path, WeTA_path):
        ''' Loads parameters (but not training state) '''
        print("loading encoder, decoder, critic, WeTA")
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.WeTA.load_state_dict(torch.load(WeTA_path))

