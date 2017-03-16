from colour import Colour


class Tree(object):

    def __init__(self, root):
        self.root = root
        self.nodes = []

    def add_node(self, label, node):
        self.nodes.append((label, node))

    def __str__(self, depth=1):
        if self.nodes:
            string = Colour.GREEN + "A" + str(self.root+1) + Colour.END
            for label, node in self.nodes:
                string += "\n" + "|\t"*depth + "\b\b" + Colour.BLUE + str(label) + Colour.END + " - "
                if type(node) is int: string += Colour.RED + str(node) + Colour.END
                else: string += node.__str__(depth+1)
            return string
        else:
            return self.root

    def decide(self, attributes):
        node = self
        while not type(node) is int:
            label, node = node.nodes[attributes[node.root]]
        return node
