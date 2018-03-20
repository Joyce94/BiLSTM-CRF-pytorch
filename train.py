import torch
import torch.nn as nn
import time
import random
import numpy as np
import data.utils as utils
import model.crf
import model.evaluation as evaluation
import data.vocab as vocab

def to_scalar(vec):
    return vec.view(-1).data.tolist()[0]

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_ner(train_insts, train_insts_index, dev_insts, dev_insts_index, test_insts, test_insts_index, model, crf_layer, config, params):
    print('training...')
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(params=parameters, lr=config.learning_rate, momentum=0.9, weight_decay=config.decay)
    best_f1 = float('-inf')

    for epoch in range(config.maxIters):
        start_time = time.time()
        model.train()
        train_insts, train_insts_index = utils.random_data(train_insts, train_insts_index)
        epoch_loss = 0
        train_buckets, train_labels_raw = params.generate_batch_buckets(config.train_batch_size, train_insts_index, char=params.add_char)

        for index in range(len(train_buckets)):
            batch_length = np.array([np.sum(mask) for mask in train_buckets[index][-1]])

            fea_v, label_v, mask_v, length_v = utils.patch_var(train_buckets[index], batch_length.tolist(), params)
            model.zero_grad()
            if mask_v.size(0) != config.train_batch_size:
                model.hidden = model.init_hidden(mask_v.size(0))
            else:
                model.hidden = model.init_hidden(config.train_batch_size)
            emit_scores = model.forward(fea_v, batch_length.tolist())

            loss = crf_layer.forward(emit_scores, label_v, mask_v)
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), config.clip_grad)
            optimizer.step()
            epoch_loss += to_scalar(loss)
        print('\nepoch is {}, average loss is {} '.format(epoch, (epoch_loss / (config.train_batch_size * len(train_buckets)))))
        # update lr
        # adjust_learning_rate(optimizer, config.learning_rate / (1 + (epoch + 1) * config.decay))
        print('Dev...')
        dev_f1 = eval(dev_insts, dev_insts_index, model, crf_layer, config, params)
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            print('\nTest...')
            test_f1 = eval(test_insts, test_insts_index, model, crf_layer, config, params)
        print('now, best fscore is: ', best_f1)


def eval(insts, insts_index, model, crf_layer, config, params):
    model.eval()
    insts, insts_index = utils.random_data(insts, insts_index)
    buckets, labels_raw = params.generate_batch_buckets(len(insts), insts_index, char=params.add_char)
    batch_length = np.array([np.sum(mask) for mask in buckets[0][-1]])
    fea_v, label_v, mask_v, length_v = utils.patch_var(buckets[0], batch_length.tolist(), params)
    model.zero_grad()
    if mask_v.size(0) != config.test_batch_size:
        model.hidden = model.init_hidden(mask_v.size(0))
    else:
        model.hidden = model.init_hidden(config.test_batch_size)
    emit_scores = model.forward(fea_v, batch_length.tolist())
    predict_path = crf_layer.viterbi_decode(emit_scores, mask_v)
    ##### predict_path: variable (seq_length, batch_size)

    # f_score = evaluation.eval_entity(label_v.data.tolist(), predict_path.transpose(0, 1).data.tolist(), params)
    f_score = evaluation.eval_entity(labels_raw[0], predict_path.transpose(0, 1).data.tolist(), params)
    return f_score






