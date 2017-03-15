import math, random, copy
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

# Initiate random importances for random_importance()
random_importances = [random.uniform(0, 1) for i in range(len(attributes))]


def decision_tree_learning(examples, attributes, parent_examples):
    if not examples: return plurality_value(parent_examples)
    elif classes_are_equal(examples): return examples[0][-1]
    elif not attributes: return plurality_value(examples)
    else:
        attribute_importances = []
        for a in attributes: attribute_importances.append(importance_random(a, examples))
        A = np.argmax(attribute_importances)

        tree = Tree(A)

        for v in [0, 1]:
            new_examples = []
            for e in examples:
                if e[A] == v: new_examples.append(copy.deepcopy(e))

            new_attributes = copy.deepcopy(attributes)
            new_attributes.remove(A)

            subtree = (decision_tree_learning(new_examples, attributes, examples))
            tree.add_branch(v, subtree)
    return tree


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


def importance_random(a, examples):
    return random_importances[a]


def importance_entropy(a, examples):
    entropy = -(0.5*math.log2(0.5))


class Tree:
    tree = []

    def __init__(self, root):
        self.tree.append(root)

    def add_branch(self, label, subtree):
        self.tree.append(subtree)

decision_tree_learning(examples, attributes, examples)
