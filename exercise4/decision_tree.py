import random
import numpy as np
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
attributes = set()
for i in range(len(examples[0]) - 1):
    attributes.add(i)
print("Attributes:", attributes)


def decision_tree_learning(examples, attributes, parent_examples):
    if not examples: return plurality_value(parent_examples)
    elif classes_are_equal(examples): return examples[0][-1]
    elif not attributes: return plurality_value(examples)
    else:

        attribute_importances = []
        for a in attributes: attribute_importances.append(importance1(a, examples))
        A = np.argmax(attribute_importances)

    tree = Tree(A)


def classes_are_equal(examples):
    class_zero = examples[0][-1]
    for i in range(1, len(examples)):
        if examples[i][-1] == class_zero: return False
    return True


def plurality_value(examples):
    class_amount = [0, 0]
    for i in range(1, len(examples)):
        class_amount[examples[i][-1]] += 1
    if class_amount[0] > class_amount[1]: return 0
    elif class_amount[0] < class_amount[1]: return 1
    return random.randint(0, 1)


def importance1(a, examples):
    return 1


def importance2(a, examples):
    return None


class Tree:
    tree = []

    def __init__(self, root):
        self.tree.append(root)

    def add(self, node):
        self.tree.append(node)

decision_tree_learning(examples, attributes, examples)
