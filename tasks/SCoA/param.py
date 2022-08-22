import argparse
import os

parser = argparse.ArgumentParser()
# the required argument
parser.add_argument('--path_type', type=str, required=True, help='planner_path, player_path, or trusted_path')
parser.add_argument('--prefix', type=str, default='debug', required=True)
parser.add_argument('--batch_size', type=int, default=25, required=True)

# load model (if need to reload model)
parser.add_argument('--start_iter', type=int, default=0, required=False)
parser.add_argument('--encoder_save_path', type=str, default='', required=False)
parser.add_argument('--decoder_save_path', type=str, default='', required=False)
parser.add_argument('--critic_save_path', type=str, default='', required=False)
parser.add_argument('--WeTA_save_path', type=str, default='', required=False)

# need data
parser.add_argument("--train_vocab", type=str, default='tasks/SCoA/data/new_vocab_NL.txt')
parser.add_argument("--trainval_vocab", type=str, default='tasks/SCoA/data/new_vocab_NL.txt')
parser.add_argument("--GLOVE_NPY", type=str, default='tasks/SCoA/data/glove.npy')
parser.add_argument("--img_features", type=str, default='img_features/ResNet-152-imagenet.tsv')
parser.add_argument('--object_label', type=str, default='tasks/SCoA/data/labels.txt')
parser.add_argument('--question_data', type=str, default='tasks/SCoA/data/question_data.json')

# other default argument
parser.add_argument('--eval_type', type=str, default='val', help='val | test | val_seen | val_unseen')
parser.add_argument('--seed', type=int, default=1, required=False)
parser.add_argument('--sentence_len', type=int, default=10, required=False)
parser.add_argument("--save_path", type=str, default='tasks/SCoA/')
parser.add_argument('--feedback', type=str, default='sample', help='teacher or sample')
parser.add_argument('--angle_feat_size', type=int, default=4)
parser.add_argument('--num_view', type=int, default=36)
parser.add_argument('--featdropout', type=float, default=0.3)
parser.add_argument('--ignoreid', type=int, default=-100)
parser.add_argument("--gamma", default=0.9, type=float)
parser.add_argument('--uncertain_threshold', type=float, default=1.0)
parser.add_argument('--blind', action='store_true', help='whether to replace the ResNet encodings with zero vectors at inference time')
parser.add_argument("--explore", default=0.3, type=float, help='explore whether asking')
parser.add_argument("--ask_reward", type=float, default=-1, help='set ask reward')
parser.add_argument("--submit", type=str, default='False', help='open sumbit')

args = parser.parse_args()
args.log_dir = args.save_path + 'argument_log/'

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

with open(args.log_dir+args.prefix+'.txt', 'w') as f:
    f.write(str(args))
