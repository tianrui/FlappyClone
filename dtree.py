import numpy as np
import pdb

eps = 1e-8

class DTree(object):
    """
    Decision tree algorithm implemented 
    by adjusting the cut based on feedback
    from the game state
    """
    def __init__(self, mins, maxes):
        self.mins = mins
        self.maxes = maxes
        self.node = DNode(mins, maxes)
        return

    def infer(self, inputs):
        self.inputs = inputs
        decision = 0
        node_i = self.node
        for i in np.arange(len(self.inputs)):
            if inputs[i] < node_i.cut and inputs[i] > self.mins[i]:
                decision = 0
                node_i = node_i.left
            elif inputs[i] >= node_i.cut and inputs[i] < self.maxes[i]:
                decision = 1
                node_i = node_i.right
            else:
                raise Exception('Invalid input value %d for input %d' % (self.inputs[i], i))

        return decision

    def feedback(self, inputs, outcome):
        """
        outcome = 1: positive
        outcome = 0: negative
        """
        for i in np.arange(len(self.mins)):
            if outcome == 0:
                if inputs[i] < self.node.cut:
                    self.node.cut = inputs[i] - eps
                if inputs[i] >= self.node.cut:
                    self.node.cut = inputs[i] + eps
        return
                   
    def save(self):
        cuts = []
        return self.node.save(cuts)
        #node_i = self.node
        #for i in np.arange(len(self.mins)):
        #    cuts.append(node_i.cut)
        #    save(


class DNode(object):
    """
    Decision tree node:
    min, max, cut
    left, right
    """
    def __init__(self, mins, maxes, cut=None):
        if len(mins) > 0 and len(maxes) > 0:
            self.min_n = mins[0]
            self.max_n = maxes[0]
            self.cut = 0
            if cut == None:
                self.cut = 0.5 * (mins[0] + maxes[0])
            else:
                self.cut = cut
            if len(mins[1:]) > 0 and len(maxes[1:]) > 0:
                self.left = DNode(mins[1:], maxes[1:])
                self.right = DNode(mins[1:], maxes[1:])
            else:
                self.left = None
                self.right = None
        return

    def save(self, cuts):
        cuts.append(self.cut)
        if self.left != None:
            cuts.append(self.left.save(cuts))
        if self.right != None:
            cuts.append(self.right.save(cuts))
        return cuts

    def load(self, cuts):
        self.cut = cuts[0]
        if self.left != None:
            cuts.append(self.left.save(cuts))
        if self.right != None:
            cuts.append(self.right.save(cuts))
        return
