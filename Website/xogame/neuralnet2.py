from __future__ import division
from numpy.random import random, randint
from random import choice
import numpy as np
from numpy import dot, array
np.random.seed(1337)


# Activation Functions
def rectified_linear(x):
    return np.maximum(0.0, x)


def tanh_sigmoid(x):
    return np.tanh(x)


def fast_sigmoid(x):
    return x / (1 + np.absolute(x))


def maxout_sigmoid(x):
    return np.maximum(0.0, np.minimum(x, 1.0))


def random_sign():
    return choice([1, -1])


def linear(x):
    return x


class NNLayer(object):
    def __init__(self, matrix, offset, activation=rectified_linear):
        self.matrix = np.array(matrix)
        self.offset = np.array(offset)
        self.activation = activation

    def __repr__(self):
        return "NNLayer(matrix={}, offset={}, activation={})".format(self.matrix.__repr__(), self.offset.__repr__(),
                                                                     self.activation)

    def calcout(self, invector):
        offset = [self.offset] * len(invector)
        line = dot(invector, self.matrix) + self.offset
        return self.activation(line)

    @classmethod
    def random(cls, inwidth, outwidth, weightrange=0.5, activation=rectified_linear):
        offset = (np.random.rand(1, outwidth) * 2 - 1) * weightrange
        weights = (np.random.rand(inwidth, outwidth) * 2 - 1) * weightrange
        return cls(weights, offset, activation=activation)


class NNLayer_Learn(NNLayer):
    def __init__(self, matrix, offset, activation=rectified_linear, maxdelta=None):
        super(NNLayer_Learn, self).__init__(matrix, offset, activation=activation)
        if maxdelta is None:
            self.maxdelta = random() * 2 - 1
        else:
            self.maxdelta = 0
            self.maxdelta += maxdelta
        self.columns = len(self.matrix[0])
        self.rows = len(self.matrix)

    def __repr__(self):
        return "NNLayer_Learn(matrix={}, offset={}, activation={}, maxdelta={})".format(self.matrix.__repr__(),
                                                                                        self.offset.__repr__(),
                                                                                        self.activation, self.maxdelta)

    def mutate(self, mutaterate=0.33):
        maxdelta = 0.001
        # find change in max delta
        maxdelta = self.error / 10
        # maxdelta changes up or down
        # self.maxdelta += dmaxdelta*random_sign()
        size = np.size(self.matrix)
        # make a copy of the matrix
        nmatrix = np.copy(self.matrix)
        noffset = np.copy(self.offset)

        mutatematrix = np.random.binomial(1, mutaterate, (self.rows, self.columns))
        delta = (np.random.rand(self.rows, self.columns) * 2 - 1) * self.maxdelta
        deltamatrix = mutatematrix * delta
        nmatrix += deltamatrix

        mutateoffset = np.random.binomial(1, mutaterate, (1, self.columns))
        delta = (np.random.rand(1, self.columns) * 2 - 1) * self.maxdelta
        deltaoffset = delta * mutateoffset
        noffset += deltaoffset

        return self.__class__(nmatrix, noffset, activation=self.activation, maxdelta=self.maxdelta)

    def breed(self, other, prob=0.33):
        proportionself = np.random.binomial(1, prob, (self.rows, self.columns))
        selfportion = proportionself * self.matrix
        proportionother = (proportionself * -1) + 1
        otherportion = proportionother * other.matrix
        nmatrix = selfportion + otherportion

        proportionself = np.random.binomial(1, prob, (1, self.columns))
        selfportion = proportionself * self.offset
        proportionother = (proportionself * -1) + 1
        otherportion = proportionother * other.offset
        noffset = selfportion + otherportion
        return self.__class__(nmatrix, noffset, activation=self.activation, maxdelta=self.maxdelta)


class Perceptron(NNLayer_Learn):
    def __repr__(self):
        return "Perceptron(matrix={}, offset={}, activation={}, maxdelta={})".format(self.matrix.__repr__(),
                                                                                     self.offset.__repr__(),
                                                                                     self.activation, self.maxdelta)

    def calcerror(self, inputs, goal):
        output = self.calcout(inputs)
        error = np.array(goal) - output
        average = np.sqrt((error ** 2).mean())
        self.error = average
        return average

    def graddecent(self, inputs, goal, learningrate=0.01, error=None, see=False):
        if error is None:
            output = self.calcout(inputs)
            error = np.array(goal) - output
        inputs = np.array(inputs).T
        deltamatrix = dot(inputs, error) * learningrate
        deltaoffset = error * learningrate
        nperceptron = self.copy()
        if see:
            print nperceptron.matrix
            print
            print deltamatrix
            print
            print dot(inputs, error) * learningrate
        nperceptron.matrix += deltamatrix
        nperceptron.offset += deltaoffset
        return nperceptron

    def calcangle(self, inputs, goal):
        output = self.calcout(inputs)
        normalgoals = np.array([go / np.sum(np.sqrt(go ** 2)) for go in goal])
        normalouts = np.array([out / np.sum(np.sqrt(out ** 2)) for out in output])
        angles = np.array([dot(out, goal.T) for out, goal in zip(normalouts, normalgoals)])
        aangles = angles.mean()
        self.angle = aangles
        return aangles

    def copy(self):
        return self.__class__(matrix=self.matrix, offset=self.offset, activation=self.activation,
                              maxdelta=self.maxdelta)

    @classmethod
    def initalize(cls, inwidth, outwidth, activation=rectified_linear, maxdelta=None):
        offset = np.zeros((1, outwidth)) + 0.5
        weights = (np.random.rand(inwidth, outwidth) * 2 - 1) / np.sqrt(inwidth)
        return cls(weights, offset, activation=activation, maxdelta=maxdelta)


class Hidden(object):
    def __init__(self, layers):
        self.layers = list(layers)

    def __repr__(self):
        return "Hidden(layers={})".format(self.layers.__repr__())

    def calcout(self, inputs):
        for layer in self.layers:
            inputs = layer.calcout(inputs)
        return inputs

    def calcerror(self, inputs, goal):
        outputs = self.calcout(inputs)
        error = np.array(goal) - outputs
        average = np.sqrt((error ** 2).mean())
        self.error = average
        for layer in self.layers:
            layer.error = self.error
        return average

    def graddecent(self, inputs, goal):
        nlayers = []
        input = np.array(inputs)
        reinputs = [inputs]
        for layer in self.layers:
            input = layer.calcout(np.array(input))
            reinputs.append(input)
        reinputs = list(reversed(reinputs))
        reinputs.pop(0)
        outputs = self.calcout(inputs)
        error = np.array(goal) - outputs
        for layer, input in zip(list(reversed(self.layers)), reinputs):
            nlayer = layer.graddecent(input, goal, error=error)
            error = dot(error, np.array(layer.matrix).T)
            nlayers.append(nlayer)
        nlayers = reversed(nlayers)
        return self.__class__(nlayers)

    def mutate(self):
        nlayers = []
        for layer in self.layers:
            nlayer = layer.mutate()
            nlayers.append(nlayer)
        return self.__class__(nlayers)

    def breed(self, other):
        nlayers = []
        for slayer, olayer in zip(self.layers, other.layers):
            nlayer = slayer.breed(olayer)
            nlayers.append(nlayer)
        return self.__class__(nlayers)

    def copy(self):
        return self.__class__(layers=self.layers)

    @classmethod
    def initalize(cls, inwidth, hiddenwidths, outwidth, activation=rectified_linear, endactivation=fast_sigmoid,
                  maxdelta=None):
        layers = []
        hiddenwidths = list(hiddenwidths)
        hiddenwidths.append(outwidth)
        layer = Perceptron.initalize(inwidth, hiddenwidths[0], activation=activation, maxdelta=maxdelta)
        layers.append(layer)
        if len(hiddenwidths) == 2:
            layer = Perceptron.initalize(hiddenwidths[0], hiddenwidths[1], activation=endactivation, maxdelta=maxdelta)
            layers.append(layer)
        elif len(hiddenwidths) > 2:
            for i in range(1, len(hiddenwidths) - 1):
                layer = Perceptron.initalize(hiddenwidths[i - 1], hiddenwidths[i], activation=activation,
                                             maxdelta=maxdelta)
                layers.append(layer)
            layer = Perceptron.initalize(hiddenwidths[-2], hiddenwidths[-1], activation=endactivation,
                                         maxdelta=maxdelta)
            layers.append(layer)
        return cls(layers)


class XOPerceptron(Perceptron):
    def __call__(self, board, side):
        self.side = side
        self.oside = 'o' if side == 'x' else 'x'
        return self.calcmove(board)

    def __repr__(self):
        return "XOPerceptron(matrix={}, offset={}, activation={}, maxdelta={})".format(self.matrix.__repr__(),
                                                                                       self.offset.__repr__(),
                                                                                       self.activation, self.maxdelta)

    def transboard(self, inputs):
        inputs = filter(lambda i: i != '\n', inputs)
        side = [1 if bit == self.side else 0 for bit in inputs]
        oside = [1 if bit == self.oside else 0 for bit in inputs]
        space = [1 if bit == '.' else 0 for bit in inputs]
        return np.array(side + oside + space)

    def calcmove(self, input):
        input = self.transboard(input)
        output = self.calcout(input)
        move = np.argmax(output.flatten())
        return move # board plays 0-8


class XOHidden(Hidden):
    def __call__(self, board, side):
        self.side = side
        self.oside = 'o' if side == 'x' else 'x'
        return self.calcmove(board)

    def __repr__(self):
        return "XOHidden(layers={})".format(self.layers.__repr__())

    def transboard(self, inputs):
        inputs = filter(lambda i: i != '\n', inputs)
        side = [1 if bit == self.side else 0 for bit in inputs]
        oside = [1 if bit == self.oside else 0 for bit in inputs]
        space = [1 if bit == '.' else 0 for bit in inputs]
        return np.array(side + oside + space)

    def calcmove(self, input):
        input = self.transboard(input)
        output = self.calcout(input)
        #         output = output - output.mean()
        move = output.flatten()
        return move # board plays 0-8

    def findmove(self, moves):
        y = np.exp(winchance * self.multi)
        s = np.sum(y)
        prob = y / s
        c = np.cumsum(prob)
        move = np.sum([c < np.random.random()])
        return move


class GeneticEvolution(object):
    def __init__(self, inputs, goal, population, cutoff):
        self.inputs = np.array(inputs)
        self.goal = np.array(goal)
        self.population = np.array(population)
        self.popsize = len(population)
        self.cutoff = int(cutoff)
        for pop in self.population:
            pop.calcerror(self.inputs, self.goal)

    def __call__(self, gens, threshold):
        for i in range(gens):
            fittest = sorted(self.population, key=lambda person: person.error)
            if threshold > fittest[0].error:
                print 'I have finished learning', i
                break
            else:
                self.learn(i)
        print fittest[0]
        print fittest[0].calcout(self.inputs[3])
        print self.goal[3]
        print fittest[0].error
        return fittest[0]

    def learn(self, i=0, prob=0.33):
        fittest = sorted(self.population, key=lambda person: person.error)
        fittest = fittest[:self.cutoff]
        childrenneeded = self.popsize - self.cutoff
        numchildren = 0
        children = []
        while numchildren != childrenneeded:
            chance = np.random.random_sample()
            if chance > prob:
                child = (choice(fittest).mutate()).breed((choice(fittest).mutate()))
            else:
                child = choice(fittest).mutate()
            children.append(child)
            numchildren += 1
        for child in children:
            child.calcerror(self.inputs, self.goal)
        self.population = children + fittest


class MutationLearning(object):
    def __init__(self, inputs, goal, genome):
        self.inputs = np.array(inputs)
        self.goal = np.array(goal)
        self.genome = genome.copy()
        self.genome.calcerror(inputs, goal)

    def __call__(self, gens, threshold):
        for i in range(gens):
            if threshold > self.genome.error:
                print 'I have finished learning', i
                break
            else:
                self.learn(i)
        return self.genome

    def learn(self, i=0):
        ngenome = self.genome.copy()
        ngenome.calcerror(self.inputs, self.goal)
        ngenome = ngenome.mutate()
        ngenome.calcerror(self.inputs, self.goal)
        if ngenome.error < self.genome.error:
            self.genome = ngenome
            print 'gen', i
            print 'genome.error', self.genome.error
            # print 'genome.angle',self.genome.angle


class GradeintDecent(object):
    def __init__(self, inputs, goal, genome):
        self.inputs = inputs
        self.goal = goal
        self.genome = genome
        self.genome.calcerror(self.inputs, self.goal)

    def __call__(self, gens, threshold):
        for i in range(gens):
            if self.genome.error > threshold:
                self.learn()
            else:
                print 'Finished Learning', i
                break
            self.genome.calcerror(self.inputs, self.goal)
        return self.genome

    def learn(self):
        for input, goal in zip(self.inputs, self.goal):
            self.genome = self.genome.graddecent([input], goal, learningrate=0.01)


class LayeredLearning(object):
    """This Learning Method Works For Two Layers Only"""

    def __init__(self, inputs, goal, genome):
        self.inputs = inputs
        self.goal = goal
        self.genome = genome

    def __call__(self, gens, threshold, method):
        for i in range(gens):
            if self.genome.calcerror(inputs, goal) < threshold:
                print 'I have finished learning', i
                break
            elif method == 1:
                self.frontlearn()
            else:
                self.backlearn()
        return self.genome

    @staticmethod
    def calcdelta(input, error):
        """meaningful"""
        learningrate = 0.01
        deltamatrix = dot(input, error) * learningrate
        deltaoffset = error * learningrate
        return deltamatrix, deltaoffset

    @staticmethod
    def remeberinputs(layers, input):
        hidden = layers[0].calcout(input)
        output = layers[1].calcout(hidden)
        return hidden, output

    @staticmethod
    def backtrain(frontPerceptron, hiddenact, input, fronterror):
        hiddenerror = dot(fronterror, frontPerceptron.matrix.T)
        hiddenerror = hiddenerror * (hiddenact > 0)
        return LayeredLearning.calcdelta(input.T, hiddenerror)

    @staticmethod
    def fronttrain(hiddenact, error):
        return LayeredLearning.calcdelta(hiddenact.T, error)

    @staticmethod
    def changelayer(layer, delta):
        nperceptron = layer.copy()
        nperceptron.matrix += delta[0]
        nperceptron.offset += delta[1]
        return nperceptron

    def frontlearn(self):
        for input, goal in zip(self.inputs, self.goal):
            layers = self.genome.layers
            frontPerceptron = layers[1]
            backPerceptron = layers[0]
            hidden, output = self.remeberinputs(layers, input)
            error = goal - output
            delta = self.fronttrain(hidden, error)
            newlayer = self.changelayer(frontPerceptron, delta)
            self.genome = self.genome.__class__([backPerceptron, newlayer])
        return self.genome

    def backlearn(self):  # fix
        for input, goal in zip(self.inputs, self.goal):
            layers = list(self.genome.layers)
            frontPerceptron = layers[1]
            backPerceptron = layers[0]
            hidden, output = self.remeberinputs(layers, input)
            error = goal - output
            input = np.array([input])
            delta = self.backtrain(frontPerceptron, hidden, input, error)
            newlayer = self.changelayer(backPerceptron, delta)
            self.genome = self.genome.__class__([newlayer, frontPerceptron])
        return self.genome