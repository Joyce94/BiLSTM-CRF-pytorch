import torch.nn as nn
from torch.autograd import Variable
import torch


def log_sum_exp(scores, label_nums):
    """
    params:
        scores: variable (batch_size, label_nums, label_nums)
        label_nums
    return:
        variable (batch_size, label_nums)
    """
    batch_size = scores.size(0)
    max_scores, max_index = torch.max(scores, dim=1)
    ##### max_index: variable (batch_size, label_nums)
    ##### max_scores: variable (batch_size, label_nums)
    # max_scores = torch.gather(scores, 1, max_index.view(-1, 1, label_nums)).view(-1, 1, label_nums)
    max_score_broadcast = max_scores.unsqueeze(1).view(batch_size, 1, label_nums).expand(batch_size, label_nums, label_nums)
    return max_scores.view(batch_size, label_nums) + torch.log(torch.sum(torch.exp(scores - max_score_broadcast), 1)).view(batch_size, label_nums)

def log_sum_exp_low_dim(scores):
    """
    params:
        scores: variable (batch_size, label_nums)
        label_nums
    return:
        variable (batch_size, label_nums)
    """
    # max_score = scores[0, argmax(scores)]
    batch_size = scores.size(0)
    label_nums = scores.size(1)
    max_score, max_index = torch.max(scores, 1)
    ##### max_score: variable (batch_size)
    max_score_broadcast = max_score.unsqueeze(1).expand(batch_size, label_nums)
    return max_score + torch.log(torch.sum(torch.exp(scores - max_score_broadcast), 1))

class CRF(nn.Module):
    def __init__(self, config, params):
        super(CRF, self).__init__()
        self.lstm_hiddens = config.lstm_hiddens
        # self.label_nums = params.label_num + 2
        self.label_nums = params.label_num
        self.label2id = params.label_alphabet.word2id
        self.use_cuda = params.use_cuda
        self.transition = torch.zeros(self.label_nums, self.label_nums)
        if self.use_cuda: self.transition = self.transition.cuda()
        self.T = nn.Parameter(self.transition)


    def forward(self, emit_scores, labels, masks):
        gold_scores = self.calc_sentences_scores(emit_scores, labels, masks)
        encode_scores = self.crf_encode(emit_scores, masks)
        return encode_scores - gold_scores


    def calc_sentences_scores(self, emit_scores, labels, masks):
        """
        params:
            emit_scores: variable (seq_length, batch_size, label_nums)
            labels: variable (batch_size, seq_length)
            masks: variable (batch_size, seq_length)
        """

        seq_length = emit_scores.size(0)
        batch_size = emit_scores.size(1)
        # emit_scores = emit_scores.transpose(0, 1)

        # ***** Part 2
        batch_length = torch.sum(masks, dim=1).long().unsqueeze(1)
        ends_index = torch.gather(labels, 1, (batch_length-1))

        # print(self.T[:, self.label2id['<pad>']].unsqueeze(0).view(1, self.label_nums))
        ends_transition = self.T[:, self.label2id['<pad>']].unsqueeze(0).expand(batch_size, self.label_nums)
        ends_scores = torch.gather(ends_transition, 1, ends_index)
        ##### ends_scores: variable (batch_size, 1)


        # ***** Part 1
        # labels = Variable(torch.LongTensor(list(map(lambda t: [self.label2id['<start>']] + list(t), labels.data.tolist()))))
        labels = list(map(lambda t: [self.label2id['<start>']] + list(t), labels.data.tolist()))
        ##### labels: list (batch_size, (seq_length+1))


        ##### labels_group: use lower dimension to map high dimension
        # labels_group = []
        # for label in labels:
        #     new = [label[id]*self.label_nums+label[id+1] for id in range(seq_length)]
        #     new = []
        #     for id in range(seq_length):
        #         new.append(label[id]*self.label_nums+label[id+1])
        #     labels_group.append(new)

        ##### optimize calculating the labels_group
        labels_group = [[label[id]*self.label_nums+label[id+1] for id in range(seq_length)] for label in labels]
        labels_group = Variable(torch.LongTensor(labels_group))
        if self.use_cuda: labels_group = labels_group.cuda()
        ##### labels_group: variable (batch_size, seq_length)

        batch_words_num = batch_size * seq_length
        emit_scores_broadcast = emit_scores.view(batch_words_num, -1).unsqueeze(1).view(batch_words_num, 1, self.label_nums).expand(batch_words_num, self.label_nums, self.label_nums)
        trans_scores_broadcast = self.T.unsqueeze(0).view(1, self.label_nums, self.label_nums).expand(batch_words_num, self.label_nums, self.label_nums)
        scores = emit_scores_broadcast + trans_scores_broadcast
        ##### scores: variable (batch_words_num, label_nums, label_nums)

        ##### error version
        ##### reasons: because before packing to 'batch_words_num' size, the size of emit_scores is (variable (seq_length, batch_size, label_nums)), in view of the problem of data storage, if you do it like this, you will achieve the different results with the correct version, although the different is small.
        # calc_total = torch.gather(scores.view(batch_size, seq_length, self.label_nums, self.label_nums).view(batch_size, seq_length, -1), 2, labels_group.view(batch_size, seq_length).unsqueeze(2).view(batch_size, seq_length, 1)).squeeze(2)
        ##### calc_total: variable (batch_size, seq_length)

        ##### correct version
        labels_group = labels_group.transpose(0, 1).contiguous()
        calc_total = torch.gather(scores.view(seq_length, batch_size, self.label_nums, self.label_nums).view(seq_length, batch_size, -1), 2, labels_group.view(seq_length, batch_size).unsqueeze(2).view(seq_length, batch_size, 1)).squeeze(2)

        ##### calc_total: variable (seq_length, batch_size)
        batch_scores = calc_total.masked_select(masks.transpose(0, 1))
        return batch_scores.sum() + ends_scores.sum()


    def crf_encode(self, emit_scores, masks):
        """
        params:
            emit_scores: variable (seq_length, batch_size, label_nums)
            masks: variable (batch_size, seq_length)
        """

        seq_length = emit_scores.size(0)
        batch_size = emit_scores.size(1)
        masks = masks.transpose(0, 1)

        # ***** Part 1
        ##### error version
        # emit_scores_broadcast = emit_scores[0].view(batch_size, self.label_nums).unsqueeze(1).view(batch_size, 1, self.label_nums).expand(batch_size, self.label_nums, self.label_nums)
        # trans_scores_broadcast = self.T[self.label2id['<start>'], :].unsqueeze(0).expand(self.label_nums, self.label_nums).unsqueeze(0).expand(batch_size, self.label_nums, self.label_nums)
        # seq_start_scores = emit_scores_broadcast + trans_scores_broadcast
        # ##### seq_start_scores: variable (batch_size, label_nums, label_nums)
        # # forward_scores = log_sum_exp(seq_start_scores, self.label_nums)       ## error
        # # print(forward_scores)
        # ##### forward_scores: variable (batch_size, label_nums)

        ##### correct version
        emit_scores_broadcast = emit_scores[0].view(batch_size, self.label_nums)
        trans_scores_broadcast = self.T[self.label2id['<start>'], :].unsqueeze(0).expand(batch_size, self.label_nums)
        forward_scores = emit_scores_broadcast + trans_scores_broadcast
        ##### forward_scores: variable (batch_size, label_nums)

        # ***** Part 2
        for id in range(1, seq_length):
            emit_scores_broadcast = emit_scores[id].view(batch_size, self.label_nums).unsqueeze(1).view(batch_size, 1, self.label_nums).expand(batch_size, self.label_nums, self.label_nums)
            trans_scores_broadcast = self.T.view(self.label_nums, self.label_nums).unsqueeze(0).expand(batch_size, self.label_nums, self.label_nums)
            forward_scores_broadcast = forward_scores.view(batch_size, self.label_nums).unsqueeze(2).expand(batch_size, self.label_nums, self.label_nums).clone()
            scores = emit_scores_broadcast + trans_scores_broadcast + forward_scores_broadcast
            ##### scores: variable (batch_size, label_nums, label_nums)

            cur_scores = log_sum_exp(scores, self.label_nums)
            ##### cur_scores: variable (batch_size, label_nums)

            mask = masks[id].unsqueeze(1).expand(batch_size, self.label_nums)
            masked_cur_scores = cur_scores.masked_select(mask)
            forward_scores.masked_scatter_(mask, masked_cur_scores)

        # ***** Part 3
        ##### Method 1: calculate end scores
        ends_trans_broadcast = self.T[:, self.label2id['<pad>']].unsqueeze(0).expand(batch_size, self.label_nums)
        ends_scores = ends_trans_broadcast + forward_scores.view(batch_size, self.label_nums)
        final_scores = log_sum_exp_low_dim(ends_scores)
        ##### final_scores: variable (batch_size)

        # ##### Method 2: calculate end scores, results are the same as Method 1
        # ends_trans_broadcast = self.T.view(self.label_nums, self.label_nums).unsqueeze(0).expand(batch_size, self.label_nums, self.label_nums)
        # forward_scores_broadcast = forward_scores.view(batch_size, self.label_nums).unsqueeze(2).expand(batch_size, self.label_nums, self.label_nums)
        # ends_scores = ends_trans_broadcast + forward_scores_broadcast
        # # print(ends_scores)
        # ends_sum_scores = log_sum_exp(ends_scores, self.label_nums)
        # final_scores = ends_sum_scores[:, self.label2id['<pad>']]
        # # print(final_scores)
        # ##### final_scores: variable (batch_size)

        return final_scores.sum()


    def viterbi_decode(self, emit_scores, masks):
        """"
        params:
            emit_scores: variable(seq_length, batch_size, label_num)
            masks: variable(batch_size, seq_length)
        """
        seq_length = emit_scores.size(0)
        batch_size = emit_scores.size(1)
        masks = masks.transpose(0, 1)

        # ***** Part 1
        emit_scores_broadcast = emit_scores[0].view(batch_size, self.label_nums)
        trans_scores_broadcast = self.T[self.label2id['<start>'], :].unsqueeze(0).expand(batch_size, self.label_nums)
        forward_scores = emit_scores_broadcast + trans_scores_broadcast
        ##### forward_scores: variable (batch_size, label_nums)

        # ***** Part 2
        back_path = []
        for id in range(1, seq_length):
            emit_scores_broadcast = emit_scores[id].view(batch_size, self.label_nums).unsqueeze(1).view(batch_size, 1, self.label_nums).expand(batch_size, self.label_nums, self.label_nums)
            trans_scores_broadcast = self.T.view(self.label_nums, self.label_nums).unsqueeze(0).expand(batch_size, self.label_nums, self.label_nums)
            forward_scores_broadcast = forward_scores.view(batch_size, self.label_nums).unsqueeze(2).expand(batch_size, self.label_nums, self.label_nums).clone()
            scores = emit_scores_broadcast + trans_scores_broadcast + forward_scores_broadcast
            ##### scores: variable (batch_size, label_nums, label_nums)
            max_scores, max_indexs = torch.max(scores, dim=1)
            ##### max_indexs: variable (batch_size, label_nums)
            ##### max_scores: variable (batch_size, label_nums)
            mask = masks[id].unsqueeze(1).expand(batch_size, self.label_nums)
            masked_cur_scores = max_scores.masked_select(mask)
            forward_scores.masked_scatter_(mask, masked_cur_scores)

            # mask = (1 - mask.long()).byte()
            # mask = 1 + (-1) * mask
            mask = (1 + (-1) * mask.long()).byte()

            max_indexs.masked_fill_(mask, self.label2id['<pad>'])
            ##### max_indexs: variable (batch_size, label_nums)
            back_path.append(max_indexs.data.tolist())
        ##### add a row for the position of stop_tags
        back_path.append([[self.label2id['<pad>']] * self.label_nums for _ in range(batch_size)])
        back_path = Variable(torch.LongTensor(back_path)).transpose(0, 1)
        if self.use_cuda: back_path = back_path.cuda()
        ##### back_path: variable (batch_size, seq_length, label_nums)


        # ***** Part 3
        ##### calculate end scores
        ends_trans_broadcast = self.T[:, self.label2id['<pad>']].unsqueeze(0).expand(batch_size, self.label_nums)
        forward_scores_broadcast = forward_scores.view(batch_size, self.label_nums)
        ends_scores = ends_trans_broadcast + forward_scores_broadcast

        max_scores, ends_max_indexs = torch.max(ends_scores, dim=1)
        ##### ends_max_indexs: variable (batch_size)

        # ***** Part 4
        ends_max_indexs_broadcast = ends_max_indexs.unsqueeze(1).expand(batch_size, self.label_nums)
        batch_length = torch.sum(masks, dim=0).long().unsqueeze(1)
        ends_position = batch_length.expand(batch_size, self.label_nums) - 1

        back_path.scatter_(1, ends_position.view(batch_size, self.label_nums).unsqueeze(1), ends_max_indexs_broadcast.contiguous().view(batch_size, self.label_nums).unsqueeze(1))
        ##### back_path: variable (batch_size, seq_length, label_nums)

        back_path = back_path.transpose(0, 1)
        ##### back_path: variable (seq_length, batch_size, label_nums)
        decode_path = Variable(torch.zeros(seq_length, batch_size))
        decode_path[-1] = ends_max_indexs
        for id in range(seq_length-2, -1, -1):
            ends_max_indexs = torch.gather(back_path[id], 1, ends_max_indexs.unsqueeze(1).view(batch_size, 1))
            decode_path[id] = ends_max_indexs
        return decode_path




