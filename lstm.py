import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
torch.manual_seed(123)
random.seed(123)

class LSTM(nn.Module):
    def __init__(self, config, params):
        super(LSTM, self).__init__()
        self.word_num = params.word_num
        self.label_num = params.label_num
        self.char_num = params.char_num

        self.id2word = params.word_alphabet.id2word
        self.word2id = params.word_alphabet.word2id
        self.padID = params.word_alphabet.word2id['<pad>']
        self.unkID = params.word_alphabet.word2id['<unk>']

        self.use_cuda = params.use_cuda
        self.add_char = params.add_char
        self.static = params.static

        self.feature_count = config.shrink_feature_thresholds
        self.word_dims = config.word_dims
        self.char_dims = config.char_dims

        self.lstm_hiddens = config.lstm_hiddens
        self.dropout_emb = nn.Dropout(p=config.dropout_emb)
        self.dropout_lstm = nn.Dropout(p=config.dropout_lstm)

        self.lstm_layers = config.lstm_layers
        self.batch_size = config.train_batch_size

        # self.embedding = nn.Embedding(self.word_num, self.word_dims, padding_idx=self.eofID)
        self.embedding = nn.Embedding(self.word_num, self.word_dims)
        self.embedding.weight.requires_grad = True
        if self.static:
            self.embedding_static = nn.Embedding(self.word_num, self.word_dims)
            self.embedding_static.weight.requires_grad = False

        # if config.pretrained_wordEmb_file != '':
        if params.pretrain_word_embedding is not None:
            # pretrain_weight = np.array(params.pretrain_word_embedding)
            # self.embedding.weight.data.copy_(torch.from_numpy(pretrain_weight))
            # pretrain_weight = np.array(params.pretrain_embed)
            pretrain_weight = torch.FloatTensor(params.pretrain_word_embedding)
            self.embedding.weight.data.copy_(pretrain_weight)

        # for id in range(self.word_dims):
        #     self.embedding.weight.data[self.eofID][id] = 0

        if params.static:
            self.lstm = nn.LSTM(self.word_dims*2, self.lstm_hiddens // 2, bidirectional=True, dropout=config.dropout_lstm)
        else:
            self.lstm = nn.LSTM(self.word_dims, self.lstm_hiddens // 2, bidirectional=True, dropout=config.dropout_lstm)

        self.hidden2label = nn.Linear(self.lstm_hiddens, self.label_num)

        self.hidden = self.init_hidden(self.batch_size)

        # self.crf = CRF.CRF(self.lstm_hiddens, params)
        # self.init_lstm(self.lstm)

    def init_hidden(self, batch_size):
        if self.use_cuda:
            return (Variable(torch.zeros(2, batch_size, self.lstm_hiddens // 2)).cuda(),
                     Variable(torch.zeros(2, batch_size, self.lstm_hiddens // 2)).cuda())
        else:
            return (Variable(torch.zeros(2, batch_size, self.lstm_hiddens // 2)),
                     Variable(torch.zeros(2, batch_size, self.lstm_hiddens // 2)))

    def forward(self, fea_v, length):
        if self.add_char:
            word_v = fea_v[0]       # [torch.LongTensor of size 5x16]
            char_v = fea_v[1]       # [torch.LongTensor of size 5x16x17]
        else: word_v = fea_v
        batch_size = word_v.size(0)
        seq_length = word_v.size(1)

        word_emb = self.embedding(word_v)
        word_emb = self.dropout_emb(word_emb)
        if self.static:
            word_static = self.embedding_static(word_v)
            word_static = self.dropout_emb(word_static)
            word_emb = torch.cat([word_emb, word_static], 2)
        # print(word_emb)         # [torch.FloatTensor of size 5x16x100]

        x = torch.transpose(word_emb, 0, 1)        # [torch.FloatTensor of size 13x5x100]
        # print(x)
        packed_words = pack_padded_sequence(x, length)
        # print(packed_words)
        lstm_out, self.hidden = self.lstm(packed_words, self.hidden)
        # print(lstm_out)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        # print(lstm_out)         # [torch.FloatTensor of size 16x5x200]

        lstm_out = self.dropout_lstm(lstm_out)      # [torch.FloatTensor of size 16x5x200]
        lstm_out = self.hidden2label(lstm_out).view(seq_length, batch_size, self.label_num)      # [torch.FloatTensor of size 16x5x15]

        return lstm_out