import math
import random
import copy
import numpy as np
from tree import Tree

training_data = open("data/training.txt", "r")

# Initiate examples
examples = []
for line in training_data:
    temp = []
    for numStr in line.strip("\n").split("\t"):
        temp.append(int(numStr) - 1)
    examples.append(temp)
print("Examples:", examples)

# Initiate attributes
attributes = []
for i in range(len(examples[0]) - 1):
    attributes.append(i)
print("Attributes:", attributes)


def decision_tree_learning(examples, attributes, parent_examples):
    if not examples: return plurality_value(parent_examples)
    elif classes_are_equal(examples): return examples[0][-1]
    elif not attributes: return plurality_value(examples)
    else:
        attribute_importances = []
        for a in attributes: attribute_importances.append(importance_random(a, examples))
        A = attributes[np.argmax(attribute_importances)]
        tree = Tree(A)

        for v in [0, 1]:
            new_examples = []
            for e in examples:
                if e[A] == v: new_examples.append(copy.deepcopy(e))

            new_attributes = copy.deepcopy(attributes)
            new_attributes.remove(A)

            node = (decision_tree_learning(new_examples, new_attributes, examples))
            tree.add_node(v, node)
        return tree


def classes_are_equal(examples):
    class_zero = examples[0][-1]
    for i in range(1, len(examples)):
        if examples[i][-1] != class_zero: return False
    return True


def plurality_value(examples):
    class_amount = [0, 0]
    for i in range(1, len(examples)):
        class_amount[examples[i][-1]] += 1
    if class_amount[0] > class_amount[1]: return 0
    elif class_amount[0] < class_amount[1]: return 1
    return random.randint(0, 1)


def importance_random(a, examples):
    return random.uniform(0, 1)


def importance_entropy(a, examples):
    numOf0 = 0
    numOf1 = 0
    for e in examples:
        if e[a] == 0: numOf0 += 1
        elif e[a] == 1: numOf1 += 1
    prob0 = numOf0/len(examples)
    prob1 = numOf1/len(examples)
    entropy = -(prob0 * math.log2(prob0))
    entropy += -(prob1 * math.log2(prob1))
    return entropy


def run_tests(decision_tree):
    test_data = open("data/test.txt", "r")
    test_attributes = []
    test_correct_classes = []
    for line in test_data:
        temp = []
        for numStr in line.strip("\n").split("\t"):
            temp.append(int(numStr) - 1)
        test_correct_classes.append(temp.pop(-1))
        test_attributes.append(temp)

    correct_decisions = 0
    for i in range(len(test_attributes)):
        if test_correct_classes[i] == decision_tree.decide(test_attributes[i]):
            correct_decisions += 1
    print(correct_decisions/len(test_correct_classes))


decision_tree = decision_tree_learning(examples, attributes, examples)
print(decision_tree)
run_tests(decision_tree)
