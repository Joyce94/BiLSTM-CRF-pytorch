import argparse
import data.config as config
from data.vocab import Data
import model.lstm as lstm
import train
import model.crf as crf
import torch
import random
import numpy as np
import time

if __name__ == '__main__':
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(666)
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config-file', default=r'C:\Users\song\Desktop\new-crf\BiLSTM-CRF-NER\examples\config.cfg')
    argparser.add_argument('--use-cuda', default=False)
    argparser.add_argument('--static', default=False, help='fix the embedding')
    argparser.add_argument('--add-char', default=False, help='add char feature')
    argparser.add_argument('--metric', default='exact', help='choose from [exact, binary, proportional]')

    # args = argparser.parse_known_args()
    args = argparser.parse_args()
    config = config.Configurable(args.config_file)

    data = Data()
    data.number_normalized = False
    data.static = args.static
    data.add_char = args.add_char
    data.use_cuda = args.use_cuda
    data.metric = args.metric

    test_time = time.time()
    train_insts, train_insts_index = data.get_instance(config.train_file, config.run_insts,
                                                       config.shrink_feature_thresholds, char=args.add_char)
    print('test getting train_insts time: ', time.time()-test_time)
    if not args.static:
        data.fix_alphabet()
    dev_insts, dev_insts_index = data.get_instance(config.dev_file, config.run_insts, config.shrink_feature_thresholds,
                                                   char=args.add_char)
    print('test getting dev_insts time: ', time.time() - test_time)

    data.fix_alphabet()
    test_insts, test_insts_index = data.get_instance(config.test_file, config.run_insts,
                                                     config.shrink_feature_thresholds, char=args.add_char)
    print('test getting test_insts time: ', time.time() - test_time)

    # train_buckets, train_labels_raw = data.generate_batch_buckets(config.train_batch_size, train_insts_index, char=args.add_char)
    # dev_buckets, dev_labels_raw = data.generate_batch_buckets(len(dev_insts), dev_insts_index, char=args.add_char)
    # test_buckets, test_labels_raw = data.generate_batch_buckets(len(test_insts), test_insts_index, char=args.add_char)

    print('test getting batch_insts time: ', time.time() - test_time)

    if config.pretrained_wordEmb_file != '':
        data.norm_word_emb = False
        data.build_word_pretrain_emb(config.pretrained_wordEmb_file, config.word_dims)
    if config.pretrained_charEmb_file != '':
        data.norm_char_emb = False
        data.build_char_pretrain_emb(config.pretrained_charEmb_file, config.char_dims)

    model = lstm.LSTM(config, data)
    print('test building model time: ', time.time() - test_time)

    crf_layer = crf.CRF(config, data)
    print('test creating crf time: ', time.time() - test_time)
    if data.use_cuda: model = model.cuda()

    train.train_ner(train_insts, train_insts_index, dev_insts, dev_insts_index, test_insts, test_insts_index, model, crf_layer, config, data)







