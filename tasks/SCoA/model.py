import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from param import args

class DynamicRNN(nn.Module):
    """
    This code is modified from batra-mlp-lab's repository.
    https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
    """
    def __init__(self, rnn_model):
        super(DynamicRNN, self).__init__()
        self.rnn_model = rnn_model

    def forward(self, seq_input, seq_lens=None, initial_state=None):
        """A wrapper over pytorch's rnn to handle sequences of variable length.

        Arguments
        ---------
        seq_input : torch.autograd.Variable
            Input sequence tensor (padded) for RNN model. (b, max_seq_len, embed_size)
        seq_lens : torch.LongTensor
            Length of sequences (b, )
        initial_state : torch.autograd.Variable
            Initial (hidden, cell) states of RNN model.

        Returns
        -------
            A single tensor of shape (batch_size, rnn_hidden_size) corresponding
            to the outputs of the RNN model at the last time step of each input
            sequence.
        """

        self.rnn_model.flatten_parameters()
        ctx, (h_n, c_n) = self.rnn_model(seq_input)

        c_t = c_n[-1]
        rnn_output = h_n[-1]
        return ctx, rnn_output, c_t


    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(lens.contiguous().view(-1), 0, descending=True)
        _, bwd_order = torch.sort(fwd_order)
        if isinstance(sorted_len, Variable):
            sorted_len = sorted_len.data
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)  # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


# decoder
class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                 dropout_ratio, feature_size=2048 + 4):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size + feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(
                feature[..., :-args.angle_feat_size])  # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1, c_1, logit, h_tilde


class Critic(nn.Module):
    def __init__(self, hidden_size, dropout_ratio):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()


class WeTA_Net(nn.Module):
    def __init__(self, input_size, dropout_ratio):
        super(WeTA_Net, self).__init__()

        self.state2value = nn.Sequential(
            nn.Linear(input_size, input_size//2), # (512, 256)
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(input_size//2, input_size//4), # (256, 128)
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(input_size//4, input_size//8), # (128, 64)
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(input_size//8, 2) #(64, 2)
        )

    def forward(self, state):
        # state (bs, input_size)
        logit = self.state2value(state) # (bs, 2)
        ask_predict = F.gumbel_softmax(logit, tau=1, hard=True) # (bs, 2)
        ask_predict = ask_predict[:, 1] # (bs, )

        return logit, ask_predict


# questions and target
class QuesTarAtt(nn.Module):
    def __init__(self, ques_dim, target_dim, hidden_dim):
        '''Initialize layer.'''
        super(QuesTarAtt, self).__init__()

        self.fc_tri = nn.Sequential(
            nn.Linear(ques_dim, hidden_dim, bias=False),
            nn.ReLU()
        )
        self.fc_tar = nn.Sequential(
            nn.Linear(target_dim, hidden_dim, bias=False),
            nn.ReLU()
        )
        self.sm = nn.Softmax(dim=-1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, questions, target):
        '''
        questions(bs, 36, ques_dim)
        target(bs, 1, target_dim)
        '''
        questions = self.fc_tri(questions) # (bs, 36, hidden_dim)
        questions = F.normalize(questions, p=1)

        target = self.fc_tar(target)  # (bs, 1, hidden_dim)
        target = F.normalize(target, p=1)

        # Get attention
        attn = torch.bmm(questions, target.permute(0, 2, 1)).permute(0, 2, 1)  # (bs, 36, 1)
        attn = attn.contiguous().view(-1, 1, 36) # (bs, 1, 36)
        attn = self.sm(attn)  # (bs, 1, 36)

        return attn


# questions and image
class QuesImgAtt(nn.Module):
    def __init__(self, ques_dim, image_dim, hidden_dim):
        '''Initialize layer.'''
        super(QuesImgAtt, self).__init__()
        self.fc_tri = nn.Sequential(
            nn.Linear(ques_dim, hidden_dim, bias=False),
            nn.ReLU()
        )
        self.fc_img = nn.Sequential(
            nn.Linear(image_dim, hidden_dim, bias=False),
            nn.ReLU()
        )
        self.sm = nn.Softmax(dim=-1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, questions, image):
        '''
        questions(bs, 36, ques_dim)
        image(bs, 36, image_dim)
        '''
        questions = self.fc_tri(questions)  # (bs, 36, hidden_dim)
        questions = F.normalize(questions, p=1)

        image = self.fc_img(image)  # (bs, 1, hidden_dim)
        image = F.normalize(image, p=1)

        # Get attention
        attn = torch.bmm(questions, image.permute(0, 2, 1)).permute(0, 2, 1)  # (bs, 36, 36)
        attn = torch.sum(attn, dim=-1, keepdim=True) # (bs, 36, 1)
        attn = attn.permute(0, 2, 1)  # (bs, 1, 36)
        attn = self.sm(attn)  # (bs, 1, 36)

        return attn


# encoder
class SCoA_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                 dropout_ratio, tokenizer, question_data):
        super(SCoA_Encoder, self).__init__()
        self.embedding_size = embedding_size  
        self.hidden_size = hidden_size  
        self.drop = nn.Dropout(p=dropout_ratio)
        self.tokenizer = tokenizer
        self.question_data = question_data
        self.padding_idx = padding_idx
        self.imgfeature_size = 2048
        self.sentence_len = args.sentence_len 
        self.word_embed = nn.Embedding(vocab_size, embedding_size, padding_idx)

        # agent
        self.dotAtt1 = QuesImgAtt(hidden_size, self.imgfeature_size+args.angle_feat_size, hidden_size // 2)
        self.dotAtt2 = QuesTarAtt(hidden_size, hidden_size, hidden_size // 2)
        # oracle
        self.dotAtt3 = QuesImgAtt(hidden_size, self.imgfeature_size, hidden_size // 2)

        self.WaTA_Net = torch.nn.KLDivLoss(size_average=False)

        self.sent_lstm = DynamicRNN(
            nn.LSTM(embedding_size, hidden_size, 1, dropout=dropout_ratio, batch_first=True)
        )

    def question2index(self, questions):
        # questions shape(36, 1, args.sentence_len)
        res = np.empty(shape=(36, 1, args.sentence_len), dtype=np.int32)
        [view, rows, cols] = res.shape
        for i in range(view):
            for j in range(rows):
                for k in range(cols):
                    res[i][j][k] = self.tokenizer.word_to_index[questions[i][j]['sen'][k]]
        return res

    def get_current_questions(self, long_id):
        ques_raw = np.empty((len(long_id), 36, 1, self.sentence_len), dtype=np.int32)  # shape(bs, 36, 1, self.sentence_len)
        ques_idx = np.empty((len(long_id), 36), dtype=np.int32) # (bs, 36)
        for i, id in enumerate(long_id):
            ques_raw[i, :, :, :] = self.question2index(self.question_data[id])
            for j in range(0, 36):
                ques_idx[i, j] = self.question_data[id][j][0]['idx']
        
        return ques_raw, ques_idx # (bs, 36, 1, self.sentence_len)

    # What To Ask
    def WaTA(self, ques_raw, target_enc, cur_img, next_img):
        '''
        :param ques_raw: (bs, 36, 1, self.sentence_len)
        :param target_enc: (bs, 1, 512)
        :param img: (bs, 36, 2052)
        :param next_img: (bs, 36, 2048)
        :return:
        '''

        batch_size = cur_img.shape[0]
        ques_emb = self.drop(self.word_embed(ques_raw))  # (bs, 36, 1, self.sentence_len, 300)
        
        ques_emb = ques_emb.contiguous().view(-1, self.sentence_len, 300)  # (bs*36*1, self.sentence_len, 300)
        ques_enc, h_n, c_n = self.sent_lstm(ques_emb) # (bs*36, n, 512)
        ques_enc = ques_enc.contiguous().view(batch_size, 36, -1, 512) # (bs, 36, n, 512)
        ques_enc = torch.mean(ques_enc, dim=-2, keepdim=False) # (bs, 36, 512)

        h_n = torch.mean(h_n.contiguous().view(batch_size, 36, 512), dim=-2, keepdim=False) # (bs, 512)
        c_n = torch.mean(c_n.contiguous().view(batch_size, 36, 512), dim=-2, keepdim=False) # (bs, 512)

        # agent: ques_enc and cur_img
        att1 = self.dotAtt1(ques_enc, cur_img) # (bs, 1, 36)
        # agent: ques_enc and target
        att2 = self.dotAtt2(ques_enc, target_enc) # (bs, 1, 36)
        att_agent = F.softmax(att1 + att2, dim=-1)

        # oracle: ques_enc and next_img
        att_oracle = self.dotAtt3(ques_enc, next_img)  # (bs, 1, 36)

        WaTA_loss = self.WaTA_Net(att_agent.log(), att_oracle)

        _, inds = torch.max(att_agent, dim=2) # inds: (bs, 1)
        ctx = torch.stack([torch.index_select(a, 0, i) for a, i in zip(ques_enc, inds.squeeze())]) # (bs, 1, 512)
        return WaTA_loss, ctx, h_n, c_n

    def decoder_word(self, target, bs_idx, questions, ques_idx):
        print("target = " + self.tokenizer.decode_sentence(target[bs_idx]))
        print("question = " + self.tokenizer.decode_sentence(questions[bs_idx][ques_idx]))

    def forward(self, obs, cur_img, next_img, tar_enc, encoder_target=False):
        '''
        :param obs:
        :param cur_img: (bs, 36, 2048+4)
        :param next_img: (bs, 36, 2048)
        :param tar_enc: (bs, 1, 512)
        :return:
        '''
        # encode target
        batch_size = len(obs)
        if encoder_target:
            tar_tensor = np.array([ob['tar_idx'] for ob in obs])
            tar_tensor = torch.from_numpy(tar_tensor)
            tar = Variable(tar_tensor, requires_grad=False).long().cuda()  # (bs, 720) -> (bs, 1)

            tar_emb = self.word_embed(tar) # (bs, 1, 300)
            tar_emb = self.drop(tar_emb)
            
            tar_enc, tar_h, tar_c = self.sent_lstm(tar_emb) # (bs, 1, 512)
            
            return tar_enc, tar_h, tar_c, tar_enc

        # encode question and answer
        else:
            cur_longid = [0] * batch_size
            for i, ob in enumerate(obs):
                cur_longid[i] = ob['scan'] + '_' + ob['viewpoint']

            cur_question_raw, _ = self.get_current_questions(cur_longid) # (bs, 36, 1, self.sentence_len)
            cur_question_raw = Variable(
                torch.from_numpy(cur_question_raw), requires_grad=False
            ).long().cuda()  # (bs, 36, 1, self.sentence_len)

            WaTA_loss, ans_ctx, ans_h_t, ans_c_t = self.WaTA(
                ques_raw=cur_question_raw, target_enc=tar_enc, cur_img=cur_img, next_img=next_img
            )

            # ans_ctx (bs, 1, 512)
            # ans_h_t (bs, 512)
            # ans_c_t (bs, 512)
            return WaTA_loss, ans_ctx, ans_h_t, ans_c_t
