import numpy as np
import collections
from torch.autograd import Variable
import torch
import random


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


def load_pretrained_emb_uniform(path, text_field_words_dict, emb_dims, norm=False, set_padding=False):
    padID = text_field_words_dict['<pad>']
    embed_dict, embed_dim = load_pretrained_emb_total(path)
    assert embed_dim == emb_dims
    alphabet_size = len(text_field_words_dict)
    pretrain_emb_size = len(embed_dict)
    # print('The number of words is ' + str(alphabet_size))
    print('The dim of pretrained embedding is ' + str(embed_dim) + '\n')

    pretrain_emb = np.zeros([alphabet_size, embed_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    scale = np.sqrt(3.0 / embed_dim)
    for index, word in enumerate(text_field_words_dict.keys()):
        if word in embed_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embed_dict[word])
            else:
                pretrain_emb[index,:] = embed_dict[word]
            perfect_match += 1
        elif word.lower() in embed_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embed_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embed_dict[word.lower()]
            case_match += 1
        else:
            if set_padding is False or index != padID:
                pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embed_dim])
            not_match += 1
    print("Embedding:\n  pretrain word:%s, alphabet word:%s, prefect match:%s, case match:%s, oov:%s, oov%%:%s"%(pretrain_emb_size, alphabet_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb


def load_pretrained_emb_avg(path, text_field_words_dict, emb_dims, norm=False, set_padding=True):
    print('Load embedding...')
    padID = text_field_words_dict['<pad>']
    embed_dict, embed_dim = load_pretrained_emb_total(path)
    assert embed_dim == emb_dims
    alphabet_size = len(text_field_words_dict)
    pretrain_emb_size = len(embed_dict)
    print('The dim of pretrained embedding is ' + str(embed_dim) + '\n')

    pretrain_emb = np.zeros([alphabet_size, embed_dim])
    perfect_match = []
    case_match = []
    not_match = []

    for index, word in enumerate(text_field_words_dict.keys()):
        if word in embed_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embed_dict[word])
            else:
                pretrain_emb[index,:] = embed_dict[word]
            perfect_match.append(index)
        elif word.lower() in embed_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embed_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embed_dict[word.lower()]
            case_match.append(index)
        else:
            not_match.append(index)

    sum_col = np.sum(pretrain_emb, axis=0) / (len(perfect_match)+len(case_match))  # 按列求和，再求平均
    for i in not_match:
        if i != padID or set_padding is False:
            pretrain_emb[i] = sum_col
    print("Embedding:\n  pretrain word:%s, alphabet word:%s, prefect match:%s, case match:%s, oov:%s, oov%%:%s"%(pretrain_emb_size, alphabet_size, len(perfect_match), len(case_match), len(not_match), (len(not_match)+0.)/alphabet_size))
    return pretrain_emb


def load_pretrained_emb_zeros(path, text_field_words_dict, emb_dims, norm=False, set_padding=False):
    # padID = text_field_words_dict['<pad>']
    embed_dict, embed_dim = load_pretrained_emb_total(path)
    assert embed_dim == emb_dims
    alphabet_size = len(text_field_words_dict)
    pretrain_emb_size = len(embed_dict)
    # print('The number of words is ' + str(alphabet_size))
    print('The dim of pretrained embedding is ' + str(embed_dim) + '\n')

    pretrain_emb = np.zeros([alphabet_size, embed_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for index, word in enumerate(text_field_words_dict.keys()):
        if word in embed_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embed_dict[word])
            else:
                pretrain_emb[index, :] = embed_dict[word]
            perfect_match += 1
        elif word.lower() in embed_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embed_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embed_dict[word.lower()]
            case_match += 1
        else:
            not_match += 1
    print("Embedding:\n  pretrain word:%s, alphabet word:%s, prefect match:%s, case match:%s, oov:%s, oov%%:%s" % (
    pretrain_emb_size, alphabet_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet_size))
    return pretrain_emb


def load_pretrained_emb_total(path):
    embed_dim = -1
    embed_dict = collections.OrderedDict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split(' ')
            if len(line_split) < 3: continue
            if embed_dim < 0: embed_dim = len(line_split) - 1
            else: assert (embed_dim == len(line_split) - 1)
            embed = np.zeros([1, embed_dim])        # 不直接赋值，也许考虑到python浅赋值的问题
            embed[:] = line_split[1:]
            embed_dict[line_split[0]] = embed
    return embed_dict, embed_dim


def sorted_instances(insts, insts_index):
    insts_length = [len(inst_index[0]) for inst_index in insts_index]
    insts_range = list(range(len(insts_index)))
    assert len(insts_length) == len(insts_range)
    length_dict = dict(zip(insts_range, insts_length))
    length_sorted = sorted(length_dict.items(), key=lambda e: e[1], reverse=True)
    perm_list = [length_sorted[i][0] for i in range(len(length_sorted))]
    insts_index_dict = dict(zip(insts_range, insts_index))
    insts_index_sorted = [insts_index_dict.get(i) for i in perm_list]
    insts_dict = dict(zip(insts_range, insts))
    insts_sorted = [insts_dict.get(i) for i in perm_list]
    return insts_sorted, insts_index_sorted


def patch_var(bucket, batch_length, params):
    # print(batch_length)
    if params.use_cuda:
        fea_var = Variable(torch.LongTensor(bucket[0])).cuda()
        label_var = Variable(torch.LongTensor(bucket[1])).cuda()
        mask_var = Variable(torch.ByteTensor(bucket[-1])).cuda()
        length_var = Variable(torch.LongTensor(batch_length)).cuda()
        if params.add_char:
            char_var = Variable(torch.LongTensor(bucket[2])).cuda()
            fea_var = [fea_var, char_var]
    else:
        fea_var = Variable(torch.LongTensor(bucket[0]))
        label_var = Variable(torch.LongTensor(bucket[1]))
        mask_var = Variable(torch.ByteTensor(bucket[-1]))
        length_var = Variable(torch.LongTensor(batch_length))
        if params.add_char:
            char_var = Variable(torch.LongTensor(bucket[2]))
            fea_var = [fea_var, char_var]
    return fea_var, label_var, mask_var, length_var


def random_data(insts, insts_index):
    insts_num = len(insts)
    # random.shuffle(insts)
    num_list = list(range(0, insts_num))
    random.shuffle(num_list)
    insts_dict = dict(zip(num_list, insts))
    insts_dict = sorted(insts_dict.items(), key=lambda item: item[0], reverse=False)
    insts_sorted = [ele[1] for ele in insts_dict]
    # print(insts_sorted)
    insts_index_dict = dict(zip(num_list, insts_index))
    insts_index_dict = sorted(insts_index_dict.items(), key=lambda item: item[0], reverse=False)
    insts_index_sorted = [ele[1] for ele in insts_index_dict]

    return insts_sorted, insts_index_sorted

def sorted_instances_index(insts_index):
    insts_length = [len(inst_index[0]) for inst_index in insts_index]
    insts_range = list(range(len(insts_index)))
    assert len(insts_length) == len(insts_range)
    length_dict = dict(zip(insts_range, insts_length))
    length_sorted = sorted(length_dict.items(), key=lambda e: e[1], reverse=True)
    perm_list = [length_sorted[i][0] for i in range(len(length_sorted))]
    insts_index_dict = dict(zip(insts_range, insts_index))
    insts_index_sorted = [insts_index_dict.get(i) for i in perm_list]

    return insts_index_sorted