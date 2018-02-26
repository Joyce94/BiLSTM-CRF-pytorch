from alphabet import Alphabet
from collections import Counter
import utils
import math

class Data():
    def __init__(self):
        self.word_alphabet = Alphabet('word')
        self.char_alphabet = Alphabet('char')
        self.label_alphabet = Alphabet('label', is_label=True)

        self.number_normalized = True
        self.norm_word_emb = False
        self.norm_char_emb = False

        self.pretrain_word_embedding = None
        self.pretrain_char_embedding = None

        self.max_char_length = 0

        self.word_num = 0
        self.char_num = 0
        self.label_num = 0


    def build_alphabet(self, word_counter, char_counter, label_counter, shrink_feature_threshold, char=False):
        for word, count in word_counter.most_common():
            if count > shrink_feature_threshold:
                # self.word_alphabet.id2word.append(word)
                # self.word_alphabet.id2count.append(count)
                # if self.number_normalized: word = utils.normalize_word(word)
                self.word_alphabet.add(word, count)
        for label, count in label_counter.most_common():
            if count > shrink_feature_threshold:
                # self.label_alphabet.id2word.append(label)
                # self.label_alphabet.id2count.append(count)
                self.label_alphabet.add(label, count)

        # another method
        # reverse = lambda x: dict(zip(x, range(len(x))))
        # self.word_alphabet.word2id = reverse(self.word_alphabet.id2word)
        # self.label_alphabet.word2id = reverse(self.label_alphabet.id2word)

        ##### check
        if len(self.word_alphabet.word2id) != len(self.word_alphabet.id2word) or len(self.word_alphabet.id2count) != len(self.word_alphabet.id2word):
            print('there are errors in building word alphabet.')
        if len(self.label_alphabet.word2id) != len(self.label_alphabet.id2word) or len(self.label_alphabet.id2count) != len(self.label_alphabet.id2word):
            print('there are errors in building label alphabet.')

        if char:
            for char, count in char_counter.most_common():
                if count > shrink_feature_threshold:
                    self.char_alphabet.add(char, count)
                    # self.char_alphabet.id2word.append(char)
                    # self.char_alphabet.id2count.append(count)
            # self.char_alphabet.word2id = reverse(self.char_alphabet.id2word)
            if len(self.char_alphabet.word2id) != len(self.char_alphabet.id2word) or len(
                    self.char_alphabet.id2count) != len(self.char_alphabet.id2word):
                print('there are errors in building char alphabet.')


    def fix_alphabet(self):
        self.word_num = self.word_alphabet.close()
        self.char_num = self.char_alphabet.close()
        self.label_num = self.label_alphabet.close()


    def get_instance(self, file, run_insts, shrink_feature_threshold, char=False, char_padding_symbol='<pad>'):
        words = []
        chars = []
        labels = []
        insts = []
        word_counter = Counter()
        char_counter = Counter()
        label_counter = Counter()
        char_length_max = 0
        count = 0
        with open(file, 'r', encoding='utf-8') as f:
            ##### if one sentence is a line, you can use the method to control instances for debug.
            # if run_insts == -1:
            #     fin_lines = f.readlines()
            # else:
            #     fin_lines = f.readlines()[:run_insts]
        # in_lines = open(file, 'r', encoding='utf-8').readlines()
            for line in f.readlines():
                if run_insts == count: break
                if len(line) > 2:
                    line = line.strip().split(' ')
                    if line[0] == 'token':
                        word = line[1]
                        if self.number_normalized: word = utils.normalize_word(word)
                        label = line[-1]
                        words.append(word)
                        labels.append(label)
                        word_counter[word] += 1        #####
                        label_counter[label] += 1

                        if char:
                            char_list = []
                            for char in word:
                                char_list.append(char)
                                char_counter[char] += 1
                            chars.append(char_list)
                            char_length = len(char_list)
                            if char_length > char_length_max: char_length_max = char_length
                            if char_length_max > self.max_char_length: self.max_char_length = char_length_max
                else:
                    if char:
                        chars_padded = []
                        for index, char_list in enumerate(chars):
                            char_number = len(char_list)
                            if char_number < char_length_max:
                                char_list = char_list + [char_padding_symbol] * (char_length_max - char_number)
                                char_counter[char_padding_symbol] += (char_length_max - char_number)
                            chars_padded.append(char_list)
                            assert (len(char_list) == char_length_max)

                        insts.append([words, chars_padded, labels])
                    else: insts.append([words, labels])
                    words = []
                    chars = []
                    labels = []
                    char_length_max = 0
                    count += 1
        if not self.word_alphabet.fix_flag:
            self.build_alphabet(word_counter, char_counter, label_counter, shrink_feature_threshold, char)
        insts_index = []

        for inst in insts:
            words_index = [self.word_alphabet.get_index(w) for w in inst[0]]
            labels_index = [self.label_alphabet.get_index(l) for l in inst[-1]]
            chars_index = []
            if char:
                # words, chars, labels = inst
                # words_index = [self.word_alphabet.get_index(w) for w in words]
                # labels_index = [self.label_alphabet.get_index(l) for l in labels]
                # char_index = []
                for char in inst[1]:
                    char_index = [self.char_alphabet.get_index(c) for c in char]
                    chars_index.append(char_index)
                insts_index.append([words_index, chars_index, labels_index])
            else:
                # words, labels = inst
                # words_index = [self.word_alphabet.get_index(w) for w in words]
                # labels_index = [self.label_alphabet.get_index(l) for l in labels]
                insts_index.append([words_index, labels_index])

        ##### sorted sentences
        insts_sorted, insts_index_sorted = utils.sorted_instances(insts, insts_index)
        return insts_sorted, insts_index_sorted


    def build_word_pretrain_emb(self, emb_path, word_dims):
        self.pretrain_word_embedding = utils.load_pretrained_emb_avg(emb_path, self.word_alphabet.word2id, word_dims, self.norm_word_emb)
        # self.pretrain_word_embedding = utils.load_embedding(emb_path, self.word_alphabet.word2id, word_dims)

    def build_char_pretrain_emb(self, emb_path, char_dims):
        self.pretrain_char_embedding = utils.load_pretrained_emb_avg(emb_path, self.char_alphabet.word2id, char_dims, self.norm_char_emb)


    def generate_batch_buckets(self, batch_size, insts, char=False):
        # insts_length = list(map(lambda t: len(t) + 1, inst[0] for inst in insts))
        # insts_length = list(len(inst[0]+1) for inst in insts)
        # if len(insts) % batch_size == 0:
        #     batch_num = len(insts) // batch_size
        # else:
        #     batch_num = len(insts) // batch_size + 1
        batch_num = int(math.ceil(len(insts) / batch_size))

        if char:
            buckets = [[[], [], [], []] for _ in range(batch_num)]
        else:
            buckets = [[[], [], []] for _ in range(batch_num)]
        max_length = 0

        labels_raw = [[] for _ in range(batch_num)]

        for id, inst in enumerate(insts):
            idx = id // batch_size
            if id % batch_size == 0: max_length = len(inst[0])
            cur_length = len(inst[0])

            buckets[idx][0].append(inst[0] + [self.word_alphabet.word2id['<pad>']] * (max_length - cur_length))
            buckets[idx][1].append(inst[-1] + [self.label_alphabet.word2id['<pad>']] * (max_length - cur_length))
            if char:
                # if id % batch_size == 0: max_char_length = len(inst[1][0])
                cur_char_length = len(inst[1][0])
                inst[1] = [(ele + [self.char_alphabet.word2id['<pad>']] * (self.max_char_length - cur_char_length)) for ele in inst[1]]
                buckets[idx][2].append((inst[1] + [[self.char_alphabet.word2id['<pad>']] * self.max_char_length] * (max_length - cur_length)))
            buckets[idx][-1].append([1] * cur_length + [0] * (max_length - cur_length))
            labels_raw[idx].append(inst[-1])
        return buckets, labels_raw

    def generate_batch_buckets_save(self, batch_size, insts, char=False):
        # insts_length = list(map(lambda t: len(t) + 1, inst[0] for inst in insts))
        # insts_length = list(len(inst[0]+1) for inst in insts)
        # if len(insts) % batch_size == 0:
        #     batch_num = len(insts) // batch_size
        # else:
        #     batch_num = len(insts) // batch_size + 1
        batch_num = int(math.ceil(len(insts) / batch_size))

        if char:
            buckets = [[[], [], [], []] for _ in range(batch_num)]
        else:
            buckets = [[[], [], []] for _ in range(batch_num)]
        max_length = 0
        for id, inst in enumerate(insts):
            idx = id // batch_size
            if id % batch_size == 0:
                max_length = len(inst[0]) + 1
            cur_length = len(inst[0])

            buckets[idx][0].append(inst[0] + [self.word_alphabet.word2id['<pad>']] * (max_length - cur_length))
            buckets[idx][1].append([self.label_alphabet.word2id['<start>']] + inst[-1] + [self.label_alphabet.word2id['<pad>']] * (max_length - cur_length - 1))
            if char:
                char_length = len(inst[1][0])
                buckets[idx][2].append((inst[1] + [[self.char_alphabet.word2id['<pad>']] * char_length] * (max_length - cur_length)))
            buckets[idx][-1].append([1] * (cur_length + 1) + [0] * (max_length - (cur_length + 1)))

        return buckets











