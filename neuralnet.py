from __future__ import division
from random import randint,random,choice
import numpy as np
from numpy import array
from math import sqrt,exp,e
from doitxo import *
from playxo import *
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
        return self.calcout(board)

    def __repr__(self):
        return "NNXO(\nihmatrix={},\nhomatrix={})".format(self.ihmatrix.__repr__(), self.homatrix.__repr__())

    # Custom Functions
    def mutate(self):
        learningrate = 0.5
        """Makes random changes to the weights"""
        # if it forfeits every game because it's stupid : make a new instance of it
        if self.error == sqrt((-40)**2/20):
            return NNXO.newrandom(self.ihcolumns-1) # fix this
        else: # else : mutate
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
        return np.array([1 if bit == self.side else -1 if bit == self.oside else 0 for bit in inputs])

    def calcout(self, inputs):
        ins = self.transboard(inputs)
        ins = np.hstack((np.array([1]),ins))
        hidden = self.calc(self.ihmatrix,ins)
        output = self.calc(self.homatrix,hidden)
        move = np.argmax(output.flatten()[1:]) + 1
        return move

    def calcerror(self, goal, outputs):
        self.error = 0
        tests = 20
        for i in range(tests//4):
            self.error += doit('x',self,stupidai,board)
        for i in range(tests//4):
            self.error += doit('o',stupidai,self,board)
        for i in range(tests//4):
            self.error += doit('x',self,oneturnai,board)
        for i in range(tests//4):
            self.error += doit('o',oneturnai,self,board)
        return self.error

    # Class Method Functions
    @classmethod
    def newrandom(cls, hiddenwidth):
        inwidth = 9
        outwidth = 9
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

if __name__ == '__main__':
    numgens = 50
    threshold = 10**(-3)
    ins = np.array([[1,0,0],[1,1,0],[1,0,1],[1,1,1]])
    goal = np.array([[1,0],[1,1],[1,1],[1,0]])
    inwidth = 2
    hiddenwidth = 6
    outwidth = 1
    popsize = 30
    pops = [NNB.random(inwidth,hiddenwidth,outwidth) for i in range(popsize)]
    cutoff = 10

    evolve = EvolutionLearning(ins,goal,popsize,cutoff,pops)
    evolve(numgens,threshold)

    numgenerations = 5000
    learnrate = 0.5
    mutaterate = 1
    threshold = 10**(-3)
    ins = np.array([[1,0,0],[1,1,0],[1,0,1],[1,1,1]])
    goal = np.array([[1,0],[1,1],[1,1],[1,0]])
    linear = False
    debug = False
    dodescent = False
    domutate = True
    inwidth = 2
    hiddenwidth = 6
    outwidth = 1

    neuralnet = NNB.random(inwidth,hiddenwidth,outwidth)
    learn = MutationLearning(ins,goal,neuralnet)
    learn(numgenerations,threshold)

    print learn

    gens = 100000
    threshold = 20
    popsize = 100
    hiddenwidth = 9
    popu = [NNXO.newrandom(hiddenwidth) for i in range(popsize)]
    cut = 50
    see = False
    xoevolve = EvolutionLearning(board,0,popsize,cut,popu)
    xoevolve(gens,threshold)

    a = NNXO(
    ihmatrix=array([[ 1., 0.75949861, -0.1213995, 0.80891578, -0.78712543, -0.3096653,
       0.65181491, 0.25447918, 0.27124825, -0.16371406],
     [ 0., -0.58588739, 0.11890077, 0.02719951, -0.40911643, -0.65787808,
       0.52456564, 0.03654007, 0.16248066, -0.1398324],
     [ 0., -0.33342206, 0.00265934, 0.59369023, -0.16137722, -0.06838877,
       0.68442446, -0.03186446, -0.35103313, 0.90596381],
     [ 0., 0.95335608, -0.03480627, 0.74866599, -0.65171805, -0.37926504,
      -0.35169683, -0.107713, -0.0751116, 0.43797069],
     [ 0., 0.20388316, 0.79782332, 0.43693951, -0.14742214, -0.96880574,
      -0.42134693, -0.42382497, 0.16043547, -0.71821082],
     [ 0., 0.75983941, 0.73666053, 0.90575016,  0.07304822, 0.37519542,
      -0.97181112, 0.05536136, 0.33601942, 0.24480379],
     [ 0., -0.78440453, 0.56593651, 0.04727056,  0.27524957, -0.18899704,
       0.60886232, 0.07481785, 0.0886891, 0.55998216],
     [ 0., 0.47937199, -0.1517894, 0.85340356,  0.36605084, 0.28346057,
       0.76948671, -0.99367287, -1.59552416, -0.3607091],
     [ 0., 0.76965864, 0.0746829, -0.9445538, -0.70123142, 0.10964746,
      -0.12029612, -0.81074587, -0.91734712, 0.16905123],
     [ 0., 0.57418743, 0.42877752, 0.8407509, 0.29983569, 0.04463868,
      -0.57926116, 0.27028997, -0.47444066, 0.2523742]]),
    homatrix=array([[ 1., 0.7440426, 0.47553976, -0.6154771, 0.1747156, -0.45666014,
       0.46532587, -0.41968529, 0.07951368, 0.38746607],
     [ 0., 0.55963329, 0.42794849, 0.17499103, -0.35650926, -0.35075737,
       0.96128681, 0.24683135, 0.00314763, 0.39566496],
     [ 0., -0.09283099, -0.31572234, -0.36648199, 0.98743853, 0.93767238,
      -0.90379281, -0.37906045, -0.94945757, 0.30405859],
     [ 0., -0.85427438, 0.90535138, 0.04608384, -0.15231195, 0.15098231,
      -0.85846905, 0.74549927, -0.3387011, 0.3359188 ],
     [ 0., 0.68900705, -0.66875107, -0.14238529, -0.59990605, 0.09577602,
      -0.3245433, -0.14566545, -0.73195934, 0.53067017],
     [ 0., 0.16820378, -0.28882269, 0.06326133, -0.04763869, 0.959598,
       0.13264105, 0.3773556, 0.7082128, -0.25486625],
     [ 0., 0.13875448, -0.63697357, 0.26274193, -0.2932159, 0.89984499,
       0.76083135, -0.03184672, -2.29094841, -0.80954151],
     [ 0., 0.86531565, 0.56896861, -0.81828098, -0.16376552, 0.07280247,
       0.5147896, 0.97018811, 0.15290253, -0.25487183],
     [ 0., 0.25262817, -0.90510454, 0.65158334, 0.67090106, 0.40924025,
       0.76701128, -0.76951471, -0.35083822, -0.24166648],
     [ 0., 0.6527557, 0.42188332, -0.36496206, 0.92518279, 0.29933691,
       0.9568718, -0.61273548, 0.96814542, 0.72086923]]))
    b= NNXO(
ihmatrix=array([[  1.00000000e+00,  -1.13998327e-01,  -4.06591818e-01,
         -4.02129370e-04,  -7.13189671e-01,  -8.05796234e-01,
          8.33918104e-01,  -4.42993667e-02,   4.80546791e-01,
         -1.05286582e-01],
       [  0.00000000e+00,   5.15865743e-01,  -8.74473482e-01,
          8.83074033e-02,   5.57889557e-01,  -9.50518621e-01,
         -2.93359357e-01,  -4.49369668e-01,   1.46030732e-01,
          8.71936393e-01],
       [  0.00000000e+00,   7.95732538e-01,   5.75866947e-01,
          1.13082781e-01,  -3.07885564e-01,   8.22663688e-01,
          2.41926029e-02,   4.89990795e-01,   6.07619260e-01,
          5.17486251e-01],
       [  0.00000000e+00,   5.73907055e-01,   6.25174524e-01,
          7.43643626e-01,   7.46276834e-04,   7.27518272e-01,
         -4.04875290e-01,  -4.70637819e-02,  -9.62620290e-01,
          9.33288292e-01],
       [  0.00000000e+00,   1.11830755e-01,   6.73737631e-01,
         -6.92726565e-01,  -9.16257355e-01,  -3.27887619e-01,
          1.69631839e-01,   7.91378657e-01,  -1.51595762e-01,
          2.49217793e-01],
       [  0.00000000e+00,  -8.23710552e-01,  -5.37412164e-01,
         -3.35934640e-01,  -9.28731662e-01,   6.96503154e-02,
          4.39434737e-01,  -3.12829770e-01,   3.95772196e-01,
          2.06999868e-01],
       [  0.00000000e+00,  -8.01073278e-01,   6.17910629e-01,
         -4.05876247e-01,  -8.21465034e-01,  -9.60729616e-01,
          6.29394443e-01,   2.46275135e-01,  -8.80342584e-01,
         -2.08560563e-01],
       [  0.00000000e+00,   4.97667801e-01,   2.88383653e-01,
          2.71129763e-01,  -7.56216288e-01,   9.71709235e-01,
         -3.38866410e-01,  -8.49593656e-01,  -1.18702946e-01,
          4.62155832e-01],
       [  0.00000000e+00,  -4.36999273e-03,   2.56938239e-01,
         -9.08408701e-01,   3.51514098e-01,  -3.95539011e-01,
         -6.14916933e-01,   3.63230415e-01,  -7.36233205e-01,
          2.97369695e-01],
       [  0.00000000e+00,  -4.17160559e-01,   7.38765008e-01,
         -3.32096562e-02,  -3.95315733e-01,   3.39345208e-01,
         -6.30673020e-01,   7.91676010e-01,  -4.30549299e-01,
          8.59091503e-02]]),
homatrix=array([[ 1.        , -0.59834548,  0.7025908 ,  0.09338779,  0.9602418 ,
        -0.54124892,  0.83094417, -0.71969693, -0.81670218,  0.98444099],
       [ 0.        ,  0.79081947, -0.06083662, -0.69008778,  0.05466784,
        -0.03241827,  0.4761779 ,  0.65873168, -0.03704541, -0.38361654],
       [ 0.        , -0.02005229, -0.16577229,  0.9402658 , -0.90738335,
         0.37939906, -0.30749219,  0.71823209, -0.08455415, -0.25567503],
       [ 0.        ,  0.93678002, -0.17694717,  0.57926229, -0.19350674,
         0.24345195,  0.64195223,  0.98421496, -0.96376323,  0.54500942],
       [ 0.        ,  0.78025239, -0.86599528,  0.64225923, -0.50364246,
        -0.19897963, -0.67475959, -0.423288  ,  0.69415729,  0.1870407 ],
       [ 0.        ,  0.10203105, -0.860188  , -0.4209786 , -0.06251314,
        -0.95316685,  0.01190975, -0.05773312, -0.52994226, -0.24543182],
       [ 0.        , -0.95895172,  0.6621402 , -0.96465222, -0.83574106,
        -0.38708747, -0.10765152, -0.76811237,  0.05563678, -0.34881974],
       [ 0.        ,  0.64729821, -0.82827455, -0.13179525,  0.53927265,
        -0.08944508,  0.82434175, -0.28722565, -0.8384779 ,  0.89330153],
       [ 0.        , -0.62885866, -0.23829511,  0.08678683, -0.57956374,
        -0.30710188, -0.16451915,  0.28287396,  0.63728198, -0.80655147],
       [ 0.        , -0.06228202, -0.04801003,  0.47335132,  0.96913166,
        -0.54748678,  0.3775301 , -0.61526925, -0.3008211 ,  0.4733433 ]]))
    see = True
    print play_game(b, stupidai, board)