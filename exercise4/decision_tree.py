import math
import random
import copy
import numpy as np
from tree import Tree
from colour import Colour


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


def importance_information_gain(a, examples):
    p = numOfPositives(examples)
    q = p/len(examples)
    return B(q) - remainder(a, examples)


def numOfPositives(examples):
    p = 0
    for e in examples:
        if e[-1] == 1: p += 1
    return p


def remainder(A, examples):
    examples0 = []
    examples1 = []
    for e in examples:
        if e[A] == 0: examples0.append(e)
        elif e[A] == 1: examples1.append(e)
    p0 = numOfPositives(examples0)
    p1 = numOfPositives(examples1)
    remainder = len(examples0)/len(examples) * B(p0/len(examples0))
    return remainder


def B(q):
    # if q == 0: return -float("inf")
    # elif q == 1: return -float("inf")
    if q == 0: return -(math.log2(1))
    elif q == 1: return -(math.log2(1))
    return -(q * math.log2(q) + (1 - q) * math.log2(1 - q))


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
    print("\nRunning tests:")
    print("Correct:", end=" ")
    for c in test_correct_classes:
        print(c, end=" ")
    print("\nActual: ", end=" ")
    for i in range(len(test_attributes)):
        correct = test_correct_classes[i]
        actual = decision_tree.decide(test_attributes[i])

        if correct == actual:
            correct_decisions += 1
            print(actual, end=" ")
        else: print(Colour.RED + str(actual) + Colour.END, end=" ")
    print("\n--------------------")
    print("{0:} {1:0.1f}%".format("Accuracy:", correct_decisions/len(test_correct_classes)*100))


decision_tree = decision_tree_learning(examples, attributes, examples)
print(decision_tree)
run_tests(decision_tree)
