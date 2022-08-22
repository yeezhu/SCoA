''' Utils for io, language, connectivity graphs etc '''

import os
import sys
import re
import string
import json
import time
import math
from collections import Counter
import numpy as np
import networkx as nx
from param import args
import torch


def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask

def new_simulator():
    import MatterSim
    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.initialize()

    return sim

def angle_feature(heading, elevation):
    import math
    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    return np.array([math.sin(heading), math.cos(heading),
                     math.sin(elevation), math.cos(elevation)] * (args.angle_feat_size // 4),
                    dtype=np.float32)

def get_point_angle_feature(baseViewId=0):
    sim = new_simulator()

    feature = np.empty((36, args.angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading

        feature[ix, :] = angle_feature(heading, state.elevation)
    return feature

def get_all_point_angle_feature():
    return [get_point_angle_feature(baseViewId) for baseViewId in range(36)]

# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>', '<NAV>', '<ORA>', '<TAR>']
padding_idx = base_vocab.index('<PAD>')

def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3], 
                                    item['pose'][7], item['pose'][11]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def load_datasets(splits):
    data = []
    for split in splits:
        assert split in ['train', 'val_seen', 'val_unseen', 'test']
        with open('tasks/SCoA/data/%s.json' % split) as f:
            data += json.load(f)
    return data


class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character
    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i

        # for prior graph
        # transform the index in tokenize to the label.txt
        '''
        self.idx_transform = {}
        labels_list = []
        with open(args.object_label, 'r') as f:
            for line in f.readlines():
                line = line.replace(' ', '').replace('\n', '').replace('\r', '')
                labels_list.append(line)

        for i, label in enumerate(labels_list):
            self.idx_transform[self.word_to_index[label]] = i
        '''


    def split_sentence(self, sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentences, seps=None):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        if type(sentences) is not list:
            sentences = [sentences]
            seps = [seps]
        for sentence, sep in zip(sentences, seps):
            if sep is not None:
                encoding.append(self.word_to_index[sep])
            for word in self.split_sentence(sentence)[::-1]:  # reverse input sentences
                if word in self.word_to_index:
                    encoding.append(self.word_to_index[word])
                else:
                    encoding.append(self.word_to_index['<UNK>'])
        encoding.append(self.word_to_index['<EOS>'])
        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length-len(encoding))

        # cut off the LHS of the encoding if it's over-size (e.g., words from the end of an individual command,
        # favoring those at the beginning of the command (since inst word order is reversed) (e.g., cut off the early
        # instructions in a dialog if the dialog is over size, preserving the latest QA pairs).
        prefix_cut = max(0, len(encoding) - self.encoding_length)
        return np.array(encoding[prefix_cut:])

    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.vocab[ix])
        return " ".join(sentence[::-1]) # unreverse before output

    def encode_dial(self, sentences, seps=None):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        if type(sentences) is list:
            for sentence, sep in zip(sentences, seps):
                if sep is not None:
                    encoding.append(self.word_to_index[sep])
                for word in self.split_sentence(sentence)[::-1]:  # reverse input sentences
                    if word in self.word_to_index:
                        encoding.append(self.word_to_index[word])
                    else:
                        encoding.append(self.word_to_index['<UNK>'])
        encoding.append(self.word_to_index['<EOS>'])
        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length-len(encoding))

        # cut off the LHS of the encoding if it's over-size (e.g., words from the end of an individual command,
        # favoring those at the beginning of the command (since inst word order is reversed) (e.g., cut off the early
        # instructions in a dialog if the dialog is over size, preserving the latest QA pairs).
        prefix_cut = max(0, len(encoding) - self.encoding_length)
        return encoding[prefix_cut:]


def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab, addLabels=True):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits)
    for item in data:
        for turn in item['dialog_history']:
            count.update(t.split_sentence(turn['message']))
    vocab = list(start_vocab)

    # Add words that are object targets.
    targets = set()
    for item in data:
        target = item['target']
        targets.add(target)
    vocab.extend(list(targets))

    # Add words above min_count threshold.
    for word, num in count.most_common():
        if word in vocab:  # targets strings may also appear as regular vocabulary.
            continue
        if num >= min_count:
            vocab.append(word)
        else:
            break

    # wy read category labels
    if addLabels:
        labels_list = []
        with open(args.object_label, 'r') as f:
            for line in f.readlines():
                line = line.replace(' ', '').replace('\n', '').replace('\r', '')
                labels_list.append(line)
            assert len(labels_list) == 142

        for word in labels_list:
            if word not in vocab:
                vocab.append(word)
    return vocab


def write_vocab(vocab, path):
    print 'Writing vocab of size %d to %s' % (len(vocab),path)
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


"""
A checkpoint manager periodically saves model and optimizer as .pth
files during training.

Checkpoint managers help with experiment reproducibility, they record
the commit SHA of your current codebase in the checkpoint saving
directory. While loading any checkpoint from other commit, they raise a
friendly warning, a signal to inspect commit diffs for potential bugs.
Moreover, they copy experiment hyper-parameters as a YAML config in
this directory.

That said, always run your experiments after committing your changes,
this doesn't account for untracked or staged, but uncommitted changes.
"""
# from pathlib import Path
from subprocess import PIPE, Popen
import warnings

import torch
from torch import nn, optim
# import yaml


class CheckpointManager(object):
    """A checkpoint manager saves state dicts of model and optimizer
    as .pth files in a specified directory. This class closely follows
    the API of PyTorch optimizers and learning rate schedulers.

    Note::
        For ``DataParallel`` modules, ``model.module.state_dict()`` is
        saved, instead of ``model.state_dict()``.

    Parameters
    ----------
    model: nn.Module
        Wrapped model, which needs to be checkpointed.
    optimizer: optim.Optimizer
        Wrapped optimizer which needs to be checkpointed.
    checkpoint_dirpath: str
        Path to an empty or non-existent directory to save checkpoints.
    step_size: int, optional (default=1)
        Period of saving checkpoints.
    last_epoch: int, optional (default=-1)
        The index of last epoch.

    Example
    --------
    >>> model = torch.nn.Linear(10, 2)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> ckpt_manager = CheckpointManager(model, optimizer, "/tmp/ckpt")
    >>> for epoch in range(20):
    ...     for batch in dataloader:
    ...         do_iteration(batch)
    ...     ckpt_manager.step()
    """

    def __init__(
        self,
        model,
        optimizer,
        checkpoint_dirpath,
    ):

        if not isinstance(model, nn.Module):
            raise TypeError("{} is not a Module".format(type(model).__name__))

        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError(
                "{} is not an Optimizer".format(type(optimizer).__name__)
            )

        self.model = model
        self.optimizer = optimizer
        self.ckpt_dirpath = checkpoint_dirpath

    def step(self, epoch, modelname):
        """Save checkpoint if step size conditions meet. """
        torch.save(
            {
                "model": self._model_state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            self.ckpt_dirpath + "checkpoint_{}_{}.pth".format(epoch, modelname)
        )


    def _model_state_dict(self):
        """Returns state dict of model, taking care of DataParallel case."""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()


def load_checkpoint(checkpoint_pthpath):
    """Given a path to saved checkpoint, load corresponding state dicts
    of model and optimizer from it. This method checks if the current
    commit SHA of codebase matches the commit SHA recorded when this
    checkpoint was saved by checkpoint manager.

    Parameters
    ----------
    checkpoint_pthpath: str or pathlib.Path
        Path to saved checkpoint (as created by ``CheckpointManager``).

    Returns
    -------
    nn.Module, optim.Optimizer
        Model and optimizer state dicts loaded from checkpoint.

    Raises
    ------
    UserWarning
        If commit SHA do not match, or if the directory doesn't have
        the recorded commit SHA.
    """
    '''
    if isinstance(checkpoint_pthpath, str):
        checkpoint_pthpath = Path(checkpoint_pthpath)
    checkpoint_dirpath = checkpoint_pthpath.resolve().parent
    checkpoint_commit_sha = list(checkpoint_dirpath.glob(".commit-*"))

    if len(checkpoint_commit_sha) == 0:
        warnings.warn(
            "Commit SHA was not recorded while saving checkpoints."
        )
    else:
        # verify commit sha, raise warning if it doesn't match
        commit_sha_subprocess = Popen(
            ["git", "rev-parse", "--short", "HEAD"], stdout=PIPE, stderr=PIPE
        )
        commit_sha, _ = commit_sha_subprocess.communicate()
        commit_sha = commit_sha.decode("utf-8").strip().replace("\n", "")

        # remove ".commit-"
        checkpoint_commit_sha = checkpoint_commit_sha[0].name[8:]

        if commit_sha != checkpoint_commit_sha:
            warnings.warn(
                f"Current commit ({commit_sha}) and the commit "
                f"({checkpoint_commit_sha}) at which checkpoint was saved,"
                " are different. This might affect reproducibility."
            )
    '''
    # load encoder, decoder, optimizer state_dicts
    components = torch.load(checkpoint_pthpath)
    return components["model"], components["optimizer"]