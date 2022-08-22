import argparse
import torch
import os
import sys
import time
import numpy as np
import pandas as pd
import json
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince
from env import R2RBatch
from model import SCoA_Encoder, Critic, WeTA_Net, AttnDecoderLSTM
from agent import SCoA_Agent
from eval import Evaluation
from param import args
from utils import CheckpointManager, load_checkpoint

TRAIN_VOCAB = args.train_vocab
TRAINVAL_VOCAB = args.trainval_vocab

prefix = args.prefix

RESULT_DIR = args.save_path + 'results/' + prefix  + '/'
SNAPSHOT_DIR = args.save_path + 'snapshots/' + prefix + '/'
PLOT_DIR = args.save_path + 'plots/' + prefix + '/'

GLOVE_NPY = args.GLOVE_NPY

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

IMAGENET_FEATURES = args.img_features

# Training settings.
agent_type = 'SCoA'

# Fixed params from MP.
features = IMAGENET_FEATURES
batch_size = args.batch_size

word_embedding_size = 256
action_embedding_size = 32
target_embedding_size = 32
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
learning_rate = 0.0001
weight_decay = 0.0005

triple_embedding_size = 300 

def train(train_env, encoder, decoder, critic, WeTA, n_iters, feedback_method, max_episode_len, model_prefix,
    log_every=100, val_envs=None):
    ''' Train on training set, validating on both seen and unseen. '''
    if val_envs is None:
        val_envs = {}

    if agent_type == 'SCoA':
        agent = SCoA_Agent(train_env, "", encoder, decoder, critic, WeTA, max_episode_len)
    else:
        sys.exit("Unrecognized agent_type '%s'" % agent_type)
    print 'Training a %s agent with %s feedback' % (agent_type, feedback_method)

    # optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate, weight_decay=weight_decay)
    WeTA_optimizer = optim.Adam(WeTA.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ckptm_encoder = CheckpointManager(encoder, encoder_optimizer, SNAPSHOT_DIR)
    ckptm_decoder = CheckpointManager(decoder, decoder_optimizer, SNAPSHOT_DIR)
    ckptm_critic = CheckpointManager(critic, critic_optimizer, SNAPSHOT_DIR)
    ckptm_WeTA = CheckpointManager(WeTA, WeTA_optimizer, SNAPSHOT_DIR)

    # load checkpoint
    if args.start_iter > 0:
        # load encoder
        enc_state_dict, enc_optimizer_state_dict = load_checkpoint(args.encoder_save_path)
        if isinstance(encoder, nn.DataParallel):
            encoder.module.load_state_dict(enc_state_dict)
        else:
            encoder.load_state_dict(enc_state_dict)
        encoder_optimizer.load_state_dict(enc_optimizer_state_dict)

        # load decoder
        dec_state_dict, dec_optimizer_state_dict = load_checkpoint(args.decoder_save_path)
        if isinstance(decoder, nn.DataParallel):
            decoder.module.load_state_dict(dec_state_dict)
        else:
            decoder.load_state_dict(dec_state_dict)
        decoder_optimizer.load_state_dict(dec_optimizer_state_dict)

        # load critic
        cri_state_dict, cri_optimizer_state_dict = load_checkpoint(args.critic_save_path)
        if isinstance(critic, nn.DataParallel):
            critic.module.load_state_dict(cri_state_dict)
        else:
            critic.load_state_dict(cri_state_dict)
        critic_optimizer.load_state_dict(cri_optimizer_state_dict)

        # load WeTA
        jug_state_dict, jug_optimizer_state_dict = load_checkpoint(args.WeTA_save_path)
        if isinstance(WeTA, nn.DataParallel):
            WeTA.module.load_state_dict(jug_state_dict)
        else:
            WeTA.load_state_dict(jug_state_dict)
        WeTA_optimizer.load_state_dict(jug_optimizer_state_dict)
        
        print("Loaded model successful")

    data_log = defaultdict(list)
    start = time.time()
                    
    for idx in range(args.start_iter, n_iters, log_every):
        interval = min(log_every,n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)

        # Train for log_every interval
        agent.train(encoder_optimizer, decoder_optimizer, critic_optimizer, WeTA_optimizer, interval, feedback=feedback_method)
        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # record Imitation loss 
        IL_losses = np.array(agent.IL_losses)
        assert len(IL_losses) == interval
        IL_loss_avg = np.average(IL_losses)
        data_log['imitation loss'].append(IL_loss_avg)
        loss_str += 'imitation loss: %.4f' % IL_loss_avg

        # record RL loss 
        RL_losses = np.array(agent.RL_losses)
        assert len(RL_losses) == interval
        RL_loss_avg = np.average(RL_losses)
        data_log['train RL loss'].append(RL_loss_avg)
        loss_str += 'RL loss: %.4f' % RL_loss_avg

        # record WeTA loss 
        WeTA_losses = np.array(agent.WeTA_losses)
        assert len(WeTA_losses) == interval
        WeTA_loss_avg = np.average(WeTA_losses)
        data_log['train WeTA loss'].append(WeTA_loss_avg)
        loss_str += 'WeTA loss: %.4f' % WeTA_loss_avg

        # record WaTA loss
        WaTA_losses = np.array(agent.WaTA_losses)
        assert len(WaTA_losses) == interval
        WaTA_loss_avg = np.average(WaTA_losses)
        data_log['train WaTA loss'].append(WaTA_loss_avg)
        loss_str += 'WaTA loss: %.4f' % WaTA_loss_avg

        torch.cuda.empty_cache()

        # Run validation
        for env_name, (env, evaluator) in val_envs.iteritems():
            agent.env = env
            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, iter)
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)

            # record imitation loss 
            IL_losses = np.array(agent.IL_losses)
            IL_losses_avg = np.average(IL_losses)
            data_log['%s imitation loss' % env_name].append(IL_losses_avg)

            # record RL loss
            val_RL_losses = np.array(agent.RL_losses)
            val_RL_loss_avg = np.average(val_RL_losses)
            data_log['%s RL loss' % env_name].append(val_RL_loss_avg)

            # record WeTA loss
            val_WeTA_losses = np.array(agent.WeTA_losses)
            val_WeTA_loss_avg = np.average(val_WeTA_losses)
            data_log['%s WeTA loss' % env_name].append(val_WeTA_loss_avg)

            # record WaTA loss
            val_WaTA_losses = np.array(agent.WaTA_losses)
            val_WaTA_loss_avg = np.average(val_WaTA_losses)
            data_log['%s WaTA loss' % env_name].append(val_WaTA_loss_avg)

            # Get validation distance from goal under test evaluation conditions
            agent.test(use_dropout=False, feedback='argmax')
            agent.write_results()
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str = ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric, val in score_summary.iteritems():
                 data_log['%s %s' % (env_name, metric)].append(val)
                 if metric in ['success_rate', 'oracle success_rate', 'oracle path_success_rate', 'dist_to_end_reduction']:
                     loss_str += ', %s: %.3f' % (metric, val)

            torch.cuda.empty_cache()

        agent.env = train_env

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str))
        df = pd.DataFrame(data_log)
        df.set_index('iteration')
        df_path = '%s%s-log.csv' % (PLOT_DIR, model_prefix)
        df.to_csv(df_path)
        
        split_string = "-".join(train_env.splits)
        enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        cri_path = '%s%s_%s_cri_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        WeTA_path = '%s%s_%s_WeTA_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)

        ckptm_encoder.step(iter, "encoder")
        ckptm_decoder.step(iter, "decoder")
        ckptm_critic.step(iter, "critic")
        ckptm_WeTA.step(iter, "WeTA")

def setup():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train', 'val_seen', 'val_unseen']), TRAINVAL_VOCAB)

def readQuestion():
    with open(args.question_data, 'r') as f:
        question_data = json.load(f)

    return question_data


def train_val(path_type, max_episode_len, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind):
    ''' Train on the training set, and validate on seen and unseen splits. '''
  
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAINVAL_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)

    question_data = readQuestion()

    # wy create glove npy
    if not os.path.exists(GLOVE_NPY):
        from init_glove import init_glove
        init_glove(tok)

    train_env = R2RBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok,
                         path_type=path_type,  blind=blind)

    # Creat validation environments
    val_envs = {split: (R2RBatch(features, batch_size=batch_size, splits=[split],
                tokenizer=tok, path_type=path_type, blind=blind),
                Evaluation([split], path_type=path_type)) for split in ['val_seen', 'val_unseen']}

    # Build models and train
    encoder = SCoA_Encoder(vocab_size=len(vocab), embedding_size=triple_embedding_size, hidden_size=hidden_size,
                         padding_idx=padding_idx, dropout_ratio=dropout_ratio, tokenizer=tok,
                         question_data=question_data).cuda()
    # read glove to init embedding
    encoder.word_embed.weight.data = torch.from_numpy(np.load(GLOVE_NPY)).cuda()
    decoder = AttnDecoderLSTM(action_embedding_size, hidden_size, dropout_ratio).cuda()
    critic = Critic(hidden_size, dropout_ratio).cuda()
    WeTA = WeTA_Net(hidden_size, dropout_ratio).cuda()

    print("start training ...")
    train(train_env, encoder, decoder, critic, WeTA, n_iters,
          feedback_method, max_episode_len, model_prefix, val_envs=val_envs)

def train_test(splits, path_type, max_episode_len, MAX_INPUT_LENGTH, feedback_method, blind):
    print "in train_test"
    setup()
    vocab = read_vocab(TRAINVAL_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)

    question_data = readQuestion()

    val_batch_size = 382
    test_env = R2RBatch(features, batch_size=val_batch_size, splits=[splits], tokenizer=tok,
                        path_type=path_type, blind=blind)

    encoder = SCoA_Encoder(vocab_size=len(vocab), embedding_size=triple_embedding_size, hidden_size=hidden_size,
                             padding_idx=padding_idx, dropout_ratio=dropout_ratio, tokenizer=tok,
                             question_data=question_data).cuda()
    decoder = AttnDecoderLSTM(action_embedding_size, hidden_size, dropout_ratio).cuda()
    critic = Critic(hidden_size, dropout_ratio).cuda()
    WeTA = WeTA_Net(hidden_size, dropout_ratio).cuda()

    agent = SCoA_Agent(test_env, "", encoder, decoder, critic, WeTA, max_episode_len)

    # agent.load(encoder_path, decoder_path, critic_path, WeTA_path)
    # load encoder
    enc_state_dict, _ = load_checkpoint(args.encoder_save_path)
    if isinstance(encoder, nn.DataParallel):
        encoder.module.load_state_dict(enc_state_dict)
    else:
        encoder.load_state_dict(enc_state_dict)

    # load decoder
    dec_state_dict, _ = load_checkpoint(args.decoder_save_path)
    if isinstance(decoder, nn.DataParallel):
        decoder.module.load_state_dict(dec_state_dict)
    else:
        decoder.load_state_dict(dec_state_dict)

    # load critic
    cri_state_dict, _ = load_checkpoint(args.critic_save_path)
    if isinstance(critic, nn.DataParallel):
        critic.module.load_state_dict(cri_state_dict)
    else:
        critic.load_state_dict(cri_state_dict)

    print "Load model successful"
    
    if not os.path.exists('tasks/SCoA/test/'):
        os.makedirs('tasks/SCoA/test/')

    agent.results_path = '%s%s_%s_%s.json' % ('tasks/SCoA/test/', args.prefix, splits, 'result')
    agent.test(use_dropout=False, feedback=feedback_method)
    agent.write_results()

    print "finish train_test"
    return

if __name__ == "__main__":

    assert args.path_type in ['planner_path', 'player_path', 'trusted_path']
    # assert args.history in ['none', 'Autogen']
    assert args.feedback in ['sample', 'teacher']
    assert args.eval_type in ['val', 'test', 'val_seen', 'val_unseen']

    # print args

    blind = args.blind

    # Set default args.
    path_type = args.path_type
    # In MP, max_episode_len = 20 while average hop range [4, 7], e.g. ~3x max.
    # max_episode_len has to account for turns; this heuristically allowed for about 1 turn per hop.
    if path_type == 'planner_path':
        max_episode_len = 20  # [1, 6], e.g., ~3x max
    else:
        max_episode_len = 80  # [2, 41], e.g., ~2x max (120 ~3x) (80 ~2x) [for player/trusted paths]

    MAX_INPUT_LENGTH = 120 * 6  

    # Training settings.
    feedback_method = args.feedback
    n_iters = 20000 # if feedback_method == 'teacher' else 20000

    # Model prefix to uniquely id this instance.
    model_prefix = '%s-SCoA-%s-%d-%s-imagenet' % (args.eval_type, path_type, max_episode_len, feedback_method)
    if blind:
        model_prefix += '-blind'

    if args.eval_type == 'val':
        train_val(path_type, max_episode_len, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind)
    elif args.eval_type == 'test':
        train_test(splits='test', path_type=path_type, max_episode_len=max_episode_len, 
                MAX_INPUT_LENGTH=MAX_INPUT_LENGTH,
                feedback_method=feedback_method, blind=blind)
    else: # can be 'val_seen', 'val_unseen'
        train_test(splits=args.eval_type, path_type=path_type, max_episode_len=max_episode_len, 
                MAX_INPUT_LENGTH=MAX_INPUT_LENGTH,
                feedback_method=feedback_method, blind=blind)

    