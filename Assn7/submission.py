'''
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
import collections
import math
import random
import util
from engine.const import Const
from util import Belief


# Class: ExactInference
# ---------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using exact updates (correct, but slow times).
class ExactInference(object):

    # Function: Init
    # --------------
    # Constructor that initializes an ExactInference object which has
    # numRows x numCols number of tiles.
    def __init__(self, numRows: int, numCols: int):
        self.skipElapse = False  ### ONLY USED BY GRADER.PY in case problem 2 has not been completed
        # util.Belief is a class (constructor) that represents the belief for a single
        # inference state of a single car (see util.py).
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()

    ##################################################################################
    # Problem 1:
    # Function: Observe (update the probabilities based on an observation)
    # -----------------
    # Takes |self.belief| -- an object of class Belief, defined in util.py --
    # and updates it in place based on the distance observation $d_t$ and
    # your position $a_t$.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard
    #                 deviation Const.SONAR_STD
    #
    # Notes:
    # - Convert row and col indices into locations using util.rowToY and util.colToX.
    # - util.pdf: computes the probability density function for a Gaussian
    # - Although the gaussian pdf is symmetric with respect to the mean and value,
    #   you should pass arguments to util.pdf in the correct order
    # - Don't forget to normalize self.belief after you update its probabilities!
    ##################################################################################

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        
        for row in range(self.belief.numRows):
            for col in range(self.belief.numCols):
                # Probability 계산에 사용되는 util 함수: 
                    # util.pdf: gaussian probability distribution function 
                        # def pdf(mean, std, value):
                
                # 거리계산에 사용되는 util 함수들: 
                    # util.colToX: return (col + 0.5) * Const.BELIEF_TILE_SIZE
                    # util.rowToY: return (row + 0.5) * Const.BELIEF_TILE_SIZE

                # Const.SOANR_STD: observed Dist 의 표준편차 
                p = util.pdf(math.sqrt((util.colToX(col) - agentX) ** 2 + (util.rowToY(row) - agentY) ** 2), Const.SONAR_STD, observedDist)
                    # p 는 현재 해당 타일 (row,col)에 차량이 있을 확률을 나타냄. 

                self.belief.setProb(row, col, self.belief.getProb(row, col) * p)
                    # 기존의 확률에 계산된 확률을 곱해서 update 하기 

        self.belief.normalize() # normalize 를 통해 모든 타일에 대한 확률의 합이 1이 됨을 확인 
        
        # END_YOUR_CODE

    ##################################################################################
    # Problem 2:
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Takes |self.belief| and updates it based on the passing of one time step.
    # Notes:
    # - Use the transition probabilities in self.transProb, which is a dictionary
    #   containing all the ((oldTile, newTile), transProb) key-val pairs that you
    #   must consider.
    # - If there are ((oldTile, newTile), transProb) pairs not in self.transProb,
    #   they are assumed to have zero probability, and you can safely ignore them.
    # - Use the addProb (or setProb) and getProb methods of the Belief class to modify
    #   and access the probabilities associated with a belief.  (See util.py.)
    # - Be careful that you are using only the CURRENT self.belief distribution to compute
    #   updated beliefs.  Don't incrementally update self.belief and use the updated value
    #   for one grid square to compute the update for another square.
    # - Don't forget to normalize self.belief after all probabilities have been updated!
    #   (so that the sum of probabilities is exactly 1 as otherwise adding/multiplying
    #    small floating point numbers can lead to sum being close to but not equal to 1)
    ##################################################################################
    def elapseTime(self) -> None:
        if self.skipElapse: ### ONLY FOR THE GRADER TO USE IN Problem 1
            return
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)

        # 새로운 belief 선언 및 초기화 -- 모든 타일의 확률이 0으로 초기화 된 상태 (기존의 belief 와 크기는 동일)
        oldBelief = util.Belief(self.belief.numRows, self.belief.numCols, 0)

        # ((oldTile, newTile), transProb) key-val pairs = transProb 형태 
        for oldTile, newTile in self.transProb:
            # 이전 타일에서의 확률, 해당 타일 쌍의 trans 확률을 곱해서 새롭게 저장 (위에서 선언한 oldBelief에 저장)
            oldBelief.addProb(newTile[0], newTile[1], self.belief.getProb(oldTile[0], oldTile[1]) * self.transProb[(oldTile, newTile)])
        
        # 1번에서 처럼 normalize 후 self 로 업데이트 (oldBelief 를 현재 belief 로 설정)
        oldBelief.normalize()
        self.belief = oldBelief

        # END_YOUR_CODE

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.
    def getBelief(self) -> Belief:
        return self.belief


# Class: Particle Filter
# ----------------------
# Maintain and update a belief distribution over the probability of a car
# being in a tile using a set of particles.
class ParticleFilter(object):
    NUM_PARTICLES = 200

    # Function: Init
    # --------------
    # Constructor that initializes an ParticleFilter object which has
    # (numRows x numCols) number of tiles.
    def __init__(self, numRows: int, numCols: int):
        self.belief = util.Belief(numRows, numCols)

        # Load the transition probabilities and store them in an integer-valued defaultdict.
        # Use self.transProbDict[oldTile][newTile] to get the probability of transitioning
        # from oldTile to newTile.
        self.transProb = util.loadTransProb()
        self.transProbDict = dict()
        for (oldTile, newTile) in self.transProb:
            if oldTile not in self.transProbDict:
                self.transProbDict[oldTile] = collections.defaultdict(int)
            self.transProbDict[oldTile][newTile] = self.transProb[(oldTile, newTile)]

        # Initialize the particles randomly.
        self.particles = collections.defaultdict(int)
        potentialParticles = list(self.transProbDict.keys())
        for _ in range(self.NUM_PARTICLES):
            particleIndex = int(random.random() * len(potentialParticles))
            self.particles[potentialParticles[particleIndex]] += 1

        self.updateBelief()

    # Function: Update Belief
    # ---------------------
    # Updates |self.belief| with the probability that the car is in each tile
    # based on |self.particles| (which is a defaultdict from grid locations to
    # number of particles at that location) and ensures that the probabilites sum to 1
    def updateBelief(self) -> None:
        newBelief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0)
        for tile in self.particles:
            newBelief.setProb(tile[0], tile[1], self.particles[tile])
        newBelief.normalize()
        self.belief = newBelief

    ##################################################################################
    # Problem 3 (part a):
    # Function: Observe:
    # -----------------
    # Takes |self.particles| and updates them based on the distance observation
    # $d_t$ and your position $a_t$.
    #
    # This algorithm takes two steps:
    # 1. Re-weight the particles based on the observation.
    #    Concept: We had an old distribution of particles, and now we want to
    #             update this particle distribution with the emission probability
    #             associated with the observed distance.
    #             Think of the particle distribution as the unnormalized posterior
    #             probability where many tiles would have 0 probability.
    #             Tiles with 0 probabilities (i.e. those with no particles)
    #             do not need to be updated.
    #             This makes particle filtering runtime to be O(|particles|).
    #             By comparison, the exact inference method (used in problem 2 + 3)
    #             assigns non-zero (though often very small) probabilities to most tiles,
    #             so the entire grid must be updated at each time step.
    # 2. Re-sample the particles.
    #    Concept: Now we have the reweighted (unnormalized) distribution, we can now
    #             re-sample the particles from this distribution, choosing a new grid location
    #             for each of the |self.NUM_PARTICLES| new particles. To be extra clear: these
    #             new NUM_PARTICLES should be sampled from the new re-weighted distribution,
    #             not the old belief distribution, with replacement so that more than
    #             one particle can be at a tile
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard deviation Const.SONAR_STD
    #
    # Notes:
    # - Remember that |self.particles| is a dictionary with keys in the form of
    #   (row, col) grid locations and values representing the number of particles at
    #   that grid square.
    # - In order to work with the grader, you must create a new dictionary when you are
    #   re-sampling the particles, then set self.particles equal to the new dictionary at the end.
    # - Create |self.NUM_PARTICLES| new particles during resampling.
    # - To pass the grader, you must call util.weightedRandomChoice() once per new
    #   particle.  See util.py for the definition of weightedRandomChoice().
    # - Although the gaussian pdf is symmetric with respect to the mean and value,
    #   you should pass arguments to util.pdf in the correct order
    ##################################################################################
    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)

        # observe 에서는 observedDist 를 바탕으로 particle distribution update 를 진행 
        old = collections.defaultdict(float)
        newParticles = collections.defaultdict(int)

        # 아래의 코드는 1번의 과정과 유사하다  
        for row, col in self.particles:
            p = util.pdf(math.sqrt((util.colToX(col) - agentX) ** 2 + (util.rowToY(row) - agentY) ** 2), Const.SONAR_STD, observedDist)
            old[(row, col)] = self.particles[(row, col)] * p
        
        # 새롭게 정의된 weighted distribution 에 대해서 새롭게 particle 을 정의한다. (weight 이 큰 particle 을 선택하도록 함)
        for i in range(self.NUM_PARTICLES):
            particle = util.weightedRandomChoice(old) # key
            newParticles[particle] += 1 # value
        
        self.particles = newParticles

        # END_YOUR_CODE

        self.updateBelief()

    ##################################################################################
    # Problem 3 (part a):
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Reads |self.particles|, representing particle locations at time $t$, and
    # writes an updated |self.particles| with particle locations at time $t+1$.
    #
    # This algorithm takes one step:
    # 1. Proposal based on the particle distribution at current time $t$.
    #    Concept: We have a particle distribution at current time $t$, and we want
    #             to propose the particle distribution at time $t+1$. We would like
    #             to sample again to see where each particle would end up using
    #             the transition model.
    #
    # Notes:
    # - Transition probabilities are stored in |self.transProbDict|.
    # - To pass the grader, you must loop over the particles using a statement
    #   of the form 'for particle in self.particles: <your code>' and call
    #   util.weightedRandomChoice() to sample a new particle location.
    # - Remember that if there are multiple particles at a particular location,
    #   you will need to call util.weightedRandomChoice() once for each of them!
    # - You should NOT call self.updateBelief() at the end of this function.
    ##################################################################################
    def elapseTime(self) -> None:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)

        # 2번에서 처럼 새로운 dict 선언 
        newParticles = collections.defaultdict(int)

        for tile, num in self.particles.items():
            # 각 타일에서 다른 타일로 이동하는 확률을 저장하고 있는 trans.ProbDict 에 있는 지 확인 
            # 즉, 현재 있는 타일 "tile" 의 transtision prob 이 계산되어있는지를 확인하는 것 
            if tile in self.transProbDict:
                for i in range(num):
                    # 각 particle 의 새 위치를 weightedRandomChoice 함수를 사용해 저장 
                    newParticles[util.weightedRandomChoice(self.transProbDict[tile])] += 1
        self.particles = newParticles
        self.updateBelief()
        # END_YOUR_CODE

    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.
    def getBelief(self) -> Belief:
        return self.belief
