from __future__ import division
from random import randint,random,choice
import numpy as np
from math import sqrt,exp,e
from doitxo import *
from stratagies import *
__author__ = 'zavidan'
linear = False


class NN(object):
    # Built-In Functions
    def __init__(self):
        pass

    def __repr__(self):
        pass

    def calcerror(self,output,goal):
        error = np.array([(goal - testout) for testout, goal in zip(output,goal)])
        self.error = sqrt(np.sum(error**2)/(error[:,1:].size))
        return self.error

    # Class Method Functions
    @classmethod
    def calc(cls,layer,inputs):
        return cls.activation(inputs.dot(layer))

    @classmethod
    def activation(cls,x):
        if linear:
            return x
        else:
            return cls.sigmoid(x)

    # Static Method Functions
    @staticmethod
    def sigmoid(x):
        # return x/(1+abs(x))
        # return np.exp(-x*5)( [1/(1+exp(-n*5)) for n in x])
        # return np.tanh(x*step_factor)
        return np.maximum(0.0,x)


class NNB(NN):
    # Built-in Functions
    def __init__(self,ihmatrix,homatrix):
        """Makes the neural network using the matrices as weights"""
        self.ihmatrix = np.array(ihmatrix)
        self.homatrix = np.array(homatrix)
        self.ihcolumns = len(self.ihmatrix[0])
        self.ihrows = len(self.ihmatrix)
        self.hocolumns = len(self.homatrix[0])
        self.horows = len(self.homatrix)
        if self.ihcolumns != self.horows:
            raise Exception('ihmatrix x: {} != homatrix self.ihrows: {}'.format(self.ihcolumns, self.horows))

    def __repr__(self):
        return 'NNB(\nihmatrix=%s,\nhomatrix=%s)'%(self.ihmatrix.__repr__(),self.homatrix.__repr__())

    # Custom Functions
    def mutate(self):
        """Makes random changes to the weights"""
        delta = (random()*2-1)*self.error*2
        # Make copy of matrix
        nihmatrix = np.copy(self.ihmatrix)
        # Get random position on matrix. Can't be first column
        ihrow = randint(0,len(nihmatrix)-1)
        ihcolumn = randint(1,len(nihmatrix[0])-1)
        # Change position's weight
        nihmatrix[ihrow][ihcolumn] += delta
        # Repeat for homatrix
        nhomatrix = np.copy(self.homatrix)
        horow = randint(0,len(nhomatrix)-1)
        hocolumn = randint(1,len(nhomatrix[0])-1)
        nhomatrix[horow][hocolumn] += delta
        return NNB(nihmatrix,nhomatrix)

    def breed(self,other,prob=0.33):
        """"""
        # Breed ihmatrix
        # Get Mixer for Self and Other
        mix = np.random.binomial(1,prob,(self.ihrows,self.ihcolumns))
        inmix = 1 - mix
        # Mix Them
        smix = mix*self.ihmatrix
        omix = inmix*other.ihmatrix
        # Add the mixes togeather
        nihmatrix = smix+omix

        # Breed homatrix
        mix = np.random.binomial(1,prob,(self.horows,self.hocolumns))
        inmix = 1 - mix
        smix = mix*self.homatrix
        omix = inmix*other.homatrix
        nhomatrix = smix+omix
        return NNB(nihmatrix,nhomatrix)

    def calcout(self,inputs):
        hidden = self.calc(self.ihmatrix,inputs)
        output = self.calc(self.homatrix,hidden)
        return output

    # Class Method Functions
    @classmethod
    def random(cls,inwidth,hiddenwidth,outwidth):
        # Input - Hidden Matrix
        ihoffset = np.array([[0]for i in range(inwidth+1)])
        ihoffset[0][0] = 1
        ihweight = np.random.rand(inwidth+1,hiddenwidth)*2-1
        ihmatrix = np.hstack((ihoffset,ihweight))
        # Hidden - Output Matrix
        hooffset = np.array([[0]for i in range(hiddenwidth+1)])
        hooffset[0][0] = 1
        howeight = np.random.rand(hiddenwidth+1,outwidth)*2-1
        homatrix = np.hstack((hooffset,howeight))
        return cls(ihmatrix,homatrix)


class NNXO(NN):
    """A neural network to play naughts & crosses"""
    # Built-in Functions
    def __init__(self,ihmatrix,homatrix):
        """Makes the neural network using the matrices as weights"""
        self.side = 'x'
        self.oside = 'o'
        self.ihmatrix = np.array(ihmatrix)
        self.homatrix = np.array(homatrix)
        self.ihcolumns = len(self.ihmatrix[0])
        self.ihrows = len(self.ihmatrix)
        self.hocolumns = len(self.homatrix[0])
        self.horows = len(self.homatrix)
        if self.ihcolumns != self.horows:
            raise Exception('ihmatrix x: {} != homatrix self.ihrows: {}'.format(self.ihcolumns, self.horows))

    def __call__(self,board,side):
        self.side = side
        self.oside = 'x' if side == 'o' else 'x'
        move = self.calcmove(board)
        return move

    def __repr__(self):
        return "NNXO(\nihmatrix={},\nhomatrix={})".format(self.ihmatrix.__repr__(), self.homatrix.__repr__())

    # Custom Functions
    def mutate(self):
        learningrate = 0.5
        """Makes random changes to the weights"""
        delta = (random()*2-1)*learningrate*2
        # Make copy of matrix
        nihmatrix = np.copy(self.ihmatrix)
        # Get random position on matrix. Can't be first column
        ihrow = randint(0,len(nihmatrix)-1)
        ihcolumn = randint(1,len(nihmatrix[0])-1)
        # Change position's weight
        nihmatrix[ihrow][ihcolumn] += delta
        # Repeat for homatrix
        nhomatrix = np.copy(self.homatrix)
        horow = randint(0,len(nhomatrix)-1)
        hocolumn = randint(1,len(nhomatrix[0])-1)
        nhomatrix[horow][hocolumn] += delta
        return NNXO(nihmatrix,nhomatrix)

    def breed(self,other,prob=0.33):
        """"""
        # Breed ihmatrix
        # Get Mixer for Self and Other
        mix = np.random.binomial(1,prob,(self.ihrows,self.ihcolumns))
        inmix = 1 - mix
        # Mix Them
        smix = mix*self.ihmatrix
        omix = inmix*other.ihmatrix
        # Add the mixes togeather
        nihmatrix = smix+omix

        # Breed homatrix
        mix = np.random.binomial(1,prob,(self.horows,self.hocolumns))
        inmix = 1 - mix
        smix = mix*self.homatrix
        omix = inmix*other.homatrix
        nhomatrix = smix+omix
        return NNXO(nihmatrix,nhomatrix)

    def transboard(self, inputs):
        inputs = filter(lambda i: i != '\n', inputs)
        side = [1 if bit == self.side else 0 for bit in inputs]
        oside = [1 if bit == self.oside else 0 for bit in inputs]
        space = [1 if bit == '.' else 0 for bit in inputs]
        return np.array(side+oside+space)

    def calcmove(self, inputs):
        ins = self.transboard(inputs)
        ins = np.hstack((np.array([1]),ins))
        hidden = self.calc(self.ihmatrix,ins)
        output = self.calc(self.homatrix,hidden)
        move = np.argmax(output.flatten()[1:]) + 1
        return move

    def calcout(self, inputs):
        boardmatrix = np.array([np.hstack((np.array([1]),self.transboard(input))) for input in inputs])
        hidden = self.calc(self.ihmatrix,boardmatrix)
        output = self.calc(self.homatrix,hidden)
        output = np.array([out[1:] for out in output]) # everything execpt the offset
        return output

    def calcerror(self, outputs, goals):
        error = outputs - goals
        self.error = sqrt(np.sum(error**2)/len(outputs))
        return self.error


    # Class Method Functions
    @classmethod
    def newrandom(cls, inwidth, hiddenwidth, outwidth):
        # Input - Hidden Matrix
        ihoffset = np.array([[0]for i in range(inwidth+1)])
        ihoffset[0][0] = 1
        ihweight = np.random.rand(inwidth+1,hiddenwidth)*2-1
        ihmatrix = np.hstack((ihoffset,ihweight))
        # Hidden - Output Matrix
        hooffset = np.array([[0]for i in range(hiddenwidth+1)])
        hooffset[0][0] = 1
        howeight = np.random.rand(hiddenwidth+1,outwidth)*2-1
        homatrix = np.hstack((hooffset,howeight))
        return cls(ihmatrix, homatrix)


class MutationLearning(object):
    def __init__(self,inputs,goal,neuralnet):
        self.goal = goal
        self.inputs = inputs

        # Create NN, Calculate output, Find error
        self.neuralnet = neuralnet
        self.out = self.neuralnet.calcout(self.inputs)
        self.error = self.neuralnet.calcerror(self.out,self.goal)

    def __call__(self,gens,threshold):
        for i in range(gens):
            self.mutatelearn()
            if self.error < threshold:
                break
        print i

    def __repr__(self):
        return '''LearningStrat(inputs={inputs},goal={goal},nuralnet={nuralnet})
        Error: {error}
        Outputs: {out}
        '''.format(inputs=self.inputs,
                   goal=self.goal,
                   nuralnet=self.neuralnet,
                   error=self.error,
                   out=self.out)

    def mutatelearn(self):
        # Mutate NN, Calculate new output, Find new error
        nneuralnet = self.neuralnet.mutate()
        nout = nneuralnet.calcout(self.inputs)
        nerror = nneuralnet.calcerror(nout,self.goal)
        # Compare old error to new error
        if self.error > nerror:
            # Replace NN, output, error
            self.neuralnet = nneuralnet
            self.out = nout
            self.error = nerror
            print self.error


class EvolutionLearning(object):
    def __init__(self,inputs,goal,popsize,cutoff,population):
        self.goal = goal
        self.inputs = inputs
        self.popsize = popsize
        self.cutoff = cutoff
        self.population = population # Organisms you start off with
        for person in self.population:
            out = person.calcout(self.inputs)
            person.calcerror(out,self.goal)

    def __call__(self,gens,threshold):
        for i in range(gens):
            self.evolvelearn()
            fitness = sorted(self.population, key=lambda person: person.error)
            if i%100 == 0:
                print i
                print fitness[0].error
            if fitness[0].error < threshold:
                break

        print i
        print fitness[0]
        print fitness[0].calcout(self.inputs)
        print fitness[0].error

    def evolvelearn(self):
        fittest = sorted(self.population, key=lambda person: person.error)
        fittest = fittest[:self.cutoff]
        childrenneeded = self.popsize - self.cutoff
        children = [(choice(fittest).mutate()).breed((choice(fittest).mutate())) for i in range(childrenneeded)]
        for child in children:
            out = child.calcout(self.inputs)
            child.calcerror(out,self.goal)
        self.population = children + fittest
