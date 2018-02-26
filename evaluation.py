
class Entity():
    def __init__(self, start, end, category):
        super(Entity, self).__init__()
        self.start = start
        self.end = end
        self.category = category

    def equal(self, entity):
        return self.start == entity.start and self.end == entity.end and self.category == entity.category

    def match(self, entity):
        span = set(range(int(self.start), int(self.end) + 1))
        entity_span = set(range(int(entity.start), int(entity.end) + 1))
        return len(span.intersection(entity_span)) and self.category == entity.category

    def propor_score(self, entity):
        span = set(range(int(self.start), int(self.end) + 1))
        entity_span = set(range(int(entity.start), int(entity.end) + 1))
        return float(len(span.intersection(entity_span))) / float(len(span))

def Extract_entity(labels, category_set, prefix_array):
    idx = 0
    ent = []
    while (idx < len(labels)):
        if (is_start_label(labels[idx], prefix_array)):
            idy = idx
            endpos = -1
            while (idy < len(labels)):
                if not is_continue(labels[idy], labels[idx], prefix_array, idy - idx):
                    endpos = idy - 1
                    break
                endpos = idy
                idy += 1
            category = cleanLabel(labels[idx], prefix_array)
            # print(category)
            entity = Entity(idx, endpos, category)
            ent.append(entity)
            idx = endpos
        idx += 1
    category_num = len(category_set)
    category_list = [e for e in category_set]
    # print(category_list)
    entity_group = []
    for i in range(category_num):
        entity_group.append([])
    # print(entity_group)
    for id, c in enumerate(category_list):
        for entity in ent:
            if entity.category == c:
                entity_group[id].append(entity)
    return set(ent), entity_group

def is_start_label(label, prefix_array):
    if len(label) < 3:
        return False
    return (label[0] in prefix_array[0]) and (label[1] == '-')

def is_continue(label, startLabel, prefix_array, distance):
    if distance == 0:
        return True
    if len(label) < 3 or label == '<pad>' or label == '<start>':
        return False
    if distance != 0 and is_start_label(label, prefix_array):
        return False
    if (startLabel[0] == 's' or startLabel[0] == 'S') and startLabel[1] == '-':
        return False
    if cleanLabel(label, prefix_array) != cleanLabel(startLabel, prefix_array):
        return False
    return True

def Extract_category(label2id, prefix_array):
    prefix = [e for ele in prefix_array for e in ele]
    category_list = []
    for key in label2id:
        if '-' in key:
            category_list.append(cleanLabel(key, prefix))
    # print(category_list)
    new_list = list(set(category_list))
    new_list.sort(key=category_list.index)

    return new_list

def cleanLabel(label, prefix_array):
    prefix = [e for ele in prefix_array for e in ele]
    if len(label) > 2 and label[1] == '-':
        if label[0] in prefix:
            return label[2:]
    return label


class Eval():
    def __init__(self, category_set, dataset_num):
        self.category_set = category_set
        self.dataset_sum = dataset_num

        self.precision_c = []
        self.recall_c = []
        self.f1_score_c = []

    def clear(self):
        self.real_num = 0
        self.predict_num = 0
        self.correct_num = 0
        self.correct_num_p = 0

    def set_eval_var(self):
        category_num = len(self.category_set)
        self.B = []
        b = list(range(4))
        for i in range(category_num + 1):
            bb = [0 for e in b]
            self.B.append(bb)

    def Exact_match(self, predict_set, gold_set):
        self.clear()
        self.gold_num = len(gold_set)
        self.predict_num = len(predict_set)
        # correct_num = 0
        for p in predict_set:
            for g in gold_set:
                if p.equal(g):
                    self.correct_num += 1
                    break
        result = (self.gold_num, self.predict_num, self.correct_num)
        return result

    def Binary_evaluate(self, predict_set, gold_set):
        self.clear()
        self.gold_num = len(gold_set)
        self.predict_num = len(predict_set)
        for p in predict_set:
            for g in gold_set:
                if p.match(g):
                    self.correct_num_p += 1
                    break
        for g in gold_set:
            for p in predict_set:
                if g.match(p):
                    self.correct_num += 1
                    break
        result = (self.gold_num, self.predict_num, self.correct_num, self.correct_num_p)
        return result

    def Propor_evaluate(self, predict_set, gold_set):
        self.clear()
        self.gold_num = len(gold_set)
        self.predict_num = len(predict_set)
        for p in predict_set:
            for g in gold_set:
                if p.match(g):
                    self.correct_num_p += p.propor_score(g)
                    break
        for g in gold_set:
            for p in predict_set:
                if g.match(p):
                    self.correct_num += g.propor_score(p)
                    break
        result = (self.gold_num, self.predict_num, self.correct_num, self.correct_num_p)
        return result

    def calc_f1_score(self, eval_type):
        category_list = [e for e in self.category_set]
        category_num = len(self.category_set)
        if eval_type == 'exact':
            result = self.get_f1_score_e(self.B[0][0], self.B[0][1], self.B[0][2])
            precision = result[0]
            recall = result[1]
            f1_score = result[2]
            for iter in range(category_num):
                result = self.get_f1_score_e(self.B[iter + 1][0], self.B[iter + 1][1], self.B[iter + 1][2])
                self.precision_c.append(result[0])
                self.recall_c.append(result[1])
                self.f1_score_c.append(result[2])
        else:
            result = self.get_f1_score(self.B[0][0], self.B[0][1], self.B[0][2], self.B[0][3])
            precision = result[0]
            recall = result[1]
            f1_score = result[2]
            for iter in range(category_num):
                result = self.get_f1_score(self.B[iter + 1][0], self.B[iter + 1][1], self.B[iter + 1][2],
                                           self.B[iter + 1][3])
                self.precision_c.append(result[0])
                self.recall_c.append(result[1])
                self.f1_score_c.append(result[2])

        print('\n(The total number of dataset: {})\n'.format(self.dataset_sum))
        print('\rEvalution - precision: {:.4f}% recall: {:.4f}% f1_score: {:.4f} ({}/{}/{})'.format((precision * 100),
                                                                                                  (recall * 100),
                                                                                                  f1_score,
                                                                                                  self.B[0][2],
                                                                                                  self.B[0][1],
                                                                                                  self.B[0][0]))
        for index in range(category_num):
            print('\r   {} - precision: {:.4f}% recall: {:.4f}% f1_score: {:.4f} ({}/{}/{})'.format(
                category_list[index], (self.precision_c[index] * 100), (self.recall_c[index] * 100),
                self.f1_score_c[index], self.B[index + 1][2], self.B[index + 1][1], self.B[index + 1][0]))
        return f1_score

    def overall_evaluate(self, predict_set, gold_set, eval_type):
        if eval_type == 'exact':
            return self.Exact_match(predict_set, gold_set)
        elif eval_type == 'binary':
            return self.Binary_evaluate(predict_set, gold_set)
        elif eval_type == 'propor':
            return self.Propor_evaluate(predict_set, gold_set)

    def eval(self, gold_labels, predict_labels, eval_type, prefix_array):
        for index in range(len(gold_labels)):
            gold_set, gold_entity_group = Extract_entity(gold_labels[index], self.category_set, prefix_array)
            predict_set, pre_entity_group = Extract_entity(predict_labels[index], self.category_set, prefix_array)
            result = self.overall_evaluate(predict_set, gold_set, eval_type)  # g,p,c
            # print(result)
            for i in range(len(result)):
                self.B[0][i] += result[i]
            for iter in range(len(self.category_set)):
                result = self.overall_evaluate(pre_entity_group[iter], gold_entity_group[iter], eval_type)
                for i in range(len(result)):
                    self.B[iter + 1][i] += result[i]
        # return self.B

    def get_f1_score_e(self, real_num, predict_num, correct_num):
        if predict_num != 0:
            precision = correct_num / predict_num
        else:
            precision = 0.0
        if real_num != 0:
            recall = correct_num / real_num
        else:
            recall = 0.0
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        result = (precision, recall, f1_score)
        return result

    def get_f1_score(self, real_num, predict_num, correct_num_r, correct_num_p):
        if predict_num != 0:
            precision = correct_num_p / predict_num
        else:
            precision = 0.0
        if real_num != 0:
            recall = correct_num_r / real_num
        else:
            recall = 0.0
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        result = (precision, recall, f1_score)
        return result

def Convert2label(path, id2label):
    labels = []
    for sent in path:
        t = []
        for ele in sent:
            # print(ele)
            l = id2label[int(ele)]
            t.append(l)
        labels.append(t)
        # labels.append([id2label[int(ele)] for ele in sent])
    return labels

def eval_entity(gold_labels, predict_labels, params):
    prefix_array = [['b', 'B', 's', 'S'], ['m', 'M', 'e', 'E']]
    # print(gold_labels)
    gold_labels = Convert2label(gold_labels, params.label_alphabet.id2word)
    predict_labels = Convert2label(predict_labels, params.label_alphabet.id2word)
    # print(len(gold_labels[0]))
    # print(len(predict_labels[0]))
    # print(gold_labels[0])
    # print(predict_labels[0])
    # print(len(gold_labels[1]))
    # print(len(predict_labels[1]))

    category_set = Extract_category(params.label_alphabet.id2word, prefix_array)
    dataset_num = len(gold_labels)
    evaluation = Eval(category_set, dataset_num)
    evaluation.set_eval_var()
    evaluation.eval(gold_labels, predict_labels, params.metric, prefix_array)
    f1_score = evaluation.calc_f1_score(params.metric)
    return f1_score
