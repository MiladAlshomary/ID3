import math
import copy
import random
import csv


class ID3Node(object):
    
    def __init__(self):
        super(ID3Node, self).__init__()
        self.attribute = '_'
        self.label = '_'
        self.children = {}
        self.items = set()

    def create_edge(self, attribute_value, child):
        self.children[attribute_value] = child

    def classify(self, example):
        if self.children == {}:
            return self.label
        else:
            attr_v = example[self.attribute]
            next_node = self.children[attr_v]
            return next_node.classify(example)

    def represent(self, level=0):
        #print(('\t' * level) + 'Attribute:' + self.attribute + ' Label:' + self.label)
        if self.attribute != '_':
            s = ('\t' * level) + 'Attribute:' + self.attribute + ' Label:' + self.label + '\n'
        else:
            s = ('\t' * level) + ' Label:' + self.label + '\n'

        for c, n in self.children.items():
           s += ('\t' * (level+1) ) + c + '\n'
           s += n.represent(level+1)

        return s

    def get_leaves(self):
        if self.children == {}:
            return [self]
        else:
            leaves = []
            for c in self.children.values():
                leaves += c.get_leaves()
            return leaves

    def missclassification_rate(self, target):
        miss_classified = list(filter(lambda x: self.label != x[target], self.items))
        return len(miss_classified)/len(self.items)


        

class ID3(object):

    def conditional_entropy(self, examples, attribute, target):
        nb_examples = len(examples)

        target_values = set(map(lambda x: x[target], examples))
        attribute_values = set(map(lambda x: x[attribute], examples))

        def cond_prob(a, b):
            x = float(len(filter(lambda i: i[attribute] == b, examples)))
            y = float(len(filter(lambda i: i[attribute] == b and i[target] == a, examples)))
            return y/x

        def prob(attr_v):
            x = float(len(filter(lambda i: i[attribute] == attr_v, examples)))
            return x/len(examples)

        def H_1(attr_v):
            return -1 * (sum(map(lambda t: cond_prob(t, attr_v) * (0 if cond_prob(t, attr_v) == 0 else math.log(cond_prob(t, attr_v)) ), target_values)))

        return sum(map(lambda a_v: prob(a_v) * H_1(a_v), attribute_values))

    def majorty_class(self, examples, target):
        target_classes = set(map(lambda x: x[target], examples))
        max_v = -1
        max_c = None
        for t_c in target_classes:
            if len(filter(lambda x: x[target] == t_c, examples)) > max_v:
                max_c = t_c

        return max_c
    
    def max_info_gain(self, examples, attributes, target):
        attributes = copy.deepcopy(attributes)
        max_entropy = 0
        max_attribute = attributes.pop()

        for attr in attributes:
            attr_entropy =self.conditional_entropy(examples, attr, target)
            if attr_entropy > max_entropy:
                max_attribute = attr

        return max_attribute


    def id3(self, examples, attributes, target):
        n = ID3Node()
        n.label = self.majorty_class(examples, target)
        n.items = examples

        target_classes = set(map(lambda x: x[target], examples))

        if len(target_classes) == 1: #All examples are from same class => pure sample
            return n

        if not attributes:
            return n


        split_attr = self.max_info_gain(examples, attributes, target)
        n.attribute = split_attr

        split_attr_vals = set(map(lambda x: x[split_attr], examples))
        local_attributes = copy.deepcopy(attributes)
        local_attributes.remove(split_attr)
        for attr_v in split_attr_vals:
            sub_examples = list(filter(lambda x: x[split_attr] == attr_v, examples))
            if sub_examples == []:
                child = ID3Node()
                child.attribute = split_attr
                child.items = set()
                child.label = self.majorty_class(sub_examples, target)
            else:
                child = self.id3(sub_examples, local_attributes, target)

            n.create_edge(attr_v, child)

        return n

    def misclassification_rate(self, tree, examples, target):
        leaf_nodes = tree.get_leaves()
        error_rate = 0
        for ln in leaf_nodes:
            mc_r = ln.missclassification_rate(target)
            error_rate += mc_r * float(len(ln.items)/len(examples))
        
        return error_rate


mushrooms = \
    [{'Color': 'red', 'Eatability': 'toxic', 'Points': 'yes', 'Size': 'small'},
    {'Color': 'brown', 'Eatability': 'eatable', 'Points': 'no', 'Size': 'small'},
    {'Color': 'brown', 'Eatability': 'eatable', 'Points': 'yes', 'Size': 'large'},
    {'Color': 'green', 'Eatability': 'eatable', 'Points': 'no', 'Size': 'small'},
    {'Color': 'red', 'Eatability': 'eatable', 'Points': 'no', 'Size': 'large'}]


def partition(lst, n): 
    division = len(lst) / float(n) 
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]

def cross_validate_id3(examples, target, k):
        random.shuffle(examples)
        k_examples = partition(examples, k)

        total_error = 0
        for k_idx, k_test in enumerate(k_examples):
            tree = ID3()

            train_list = [element for i, element in enumerate(k_examples) if i != k_idx] #Train data is all folds except the test fold
            train = reduce(lambda a, b: a+b, train_list)
            node = tree.id3(train, set(['Color', 'Points', 'Size']), target)
            total_error += tree.misclassification_rate(node, k_test, target)

        return total_error/k



def load_data(data_path):
    data = []
    reader = csv.DictReader(open(data_path, 'rb'))
    for line in reader:
        data.append(line)

    return data

#data = load_data('car.data.csv')
#print(data[0])
#print(cross_validate_id3(data, 'safty', 2))

tree = ID3()
print(tree.conditional_entropy(mushrooms, 'Color', 'Eatability'))
node = tree.id3(mushrooms, set(['Color', 'Points', 'Size']), 'Eatability')
print(node.represent())
print(tree.misclassification_rate(node, mushrooms, 'Eatability'))
