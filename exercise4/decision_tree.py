

def decision_tree_learning(examples, attributes, parent_examples):
    if not examples: return plurality_value(parent_examples)
    elif not attributes: return plurality_value(examples)
    else:
        A_list = []
        for a in attributes:
            A_list.append(importance1(a, examples))
        A = max(importance1(a, examples))
        # ADD A AS ROOT IN TREE


def plurality_value(examples):
    return None


def importance1(a, examples):
    return None


def importance2(a, examples):
    return None
