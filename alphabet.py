from collections import Counter
import collections

class Alphabet():
    def __init__(self, name, is_label=False, fix_flag=False):
        self.name = name
        self.is_label = is_label
        self.word2id = collections.OrderedDict()
        self.id2word = []
        self.id2count = []
        self.fix_flag = fix_flag

        self.UNKNOWN = '<unk>'
        self.START = '<start>'
        self.PAD = '<pad>'
        if not self.is_label: self.add(self.UNKNOWN)
        if is_label: self.add(self.START)
        self.add(self.PAD)
        # print(self.word2id)
        # print(self.word2id[self.PAD])

    def add(self, word, count=-1):
        if word not in self.word2id:
            self.word2id[word] = self.size()                #####
            self.id2word.append(word)
            self.id2count.append(count)
            # print(self.word2id)
            # print(self.id2word)

    def size(self):
        return len(self.id2word)

    def get_index(self, word):
        try:
            return self.word2id[word]
        except KeyError:  # keyerror一般是使用字典里不存在的key产生的错误，避免产生这种错误
            if not self.fix_flag:
                # print('WARNING:Alphabet get_index, unknown instance, add new instance.')
                self.add(word)
                return self.word2id[word]
            else:
                # print('WARNING:Alphabet get_index, unknown instance, return unknown index.')
                return self.word2id[self.UNKNOWN]

    def close(self):
        self.fix_flag = True
        # alphabet_size = self.size()
        return self.size()             #####









