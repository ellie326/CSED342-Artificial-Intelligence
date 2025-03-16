from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from util import manhattanDistance


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()    #현재 게임 상태에서 모든 가능한 움직임의 리스트 

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]    #위의 움직임 리스트 각각에 대해서 evaluation function 결과 값을 리스트로 저장 
    bestScore = max(scores)   #위의 결과 값들 중에 제일 max 를 bestScore 에 저장 (가장 유리한 움직임)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]   #score 리스트들 중에 best Score 를 가진 node들을 bestIndices 에 리스트로 저장 
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best   #위에서 최고 점수 움직임이 여러개면 그중에 아무거나 하나 결정 

    return legalMoves[chosenIndex]    #최고 점수를 가진 state 로 이동하는 움직임을 반환. -> 즉, 최적의 행동 반환 

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    
    #Vmax 는 best value 를 말함. -infinity 로 initialize 
    vmax = float("-inf")

    #정지 행동만 포함하는 move 리스트 
    move = [Directions.STOP]

    #현재 index 에서 가능한 모든 action 에 대한 리스트 (pacman 은 index 가 0임)
    actions = gameState.getLegalActions(self.index)

    #가능한 모든 action 각각 에 대한 for loop 
    for action in actions:

      # 해당 action 에 대한 value 계산 (아래의 getQ 함수 사용)
      v = self.getQ(gameState, action)

      # max value 구하기 위해서 아래의 조건 확인 후 vmax 업데이트 
      if v > vmax:
        vmax = v
        
        # vmax 가 업데이트 될때마다 해당 action을 현재 행동으로 업데이트 (추가 x)
        move = [action]
      
      # vmax 가 여러 action 에서 나올 때 해당 action 을 move 에 추가 
      elif v == vmax:
        move.append(action)
    #print("minimax", vmax)
    # 위에서 나온 vmax 를 가지는 move 들 중에 랜덤하게 하나 채택 
    return random.choice(move)
  
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    
    # 팩맨 (index = 0) 은 max value 를 가질 수 있도록 움직이고,
    # 유령 (index != 0)들은 min value 를 가지도록 움직여야 함. 

    # max value를 반환하는 함수 -> index = 0 
    def max_value(state, depth):
        
        # 게임 종료 조건 확인 
        if state.isWin() or state.isLose():
            return state.getScore()
        
        # 깊이가 0 인 것은 더이상 미래의 상태를 탐색하지 x 
        if depth == 0:
            return self.evaluationFunction(state)
        
        LegalActions = state.getLegalActions(0)
        
        if len(LegalActions) == 0:
            return self.evaluationFunction(state)
        
        # 현재 상태에서 팩맨의 가능한 모든 움직임에 대해서 successor state들의 list 를 pacman state 에 저장 
        pacmanState = [state.generateSuccessor(0, LegalAction) for LegalAction in LegalActions]
        
        # value initialize 
        v = float("-inf")

        # 팩맨이 이동할 수 있는 모든 state들에 대해서 loop 
        for newState in pacmanState:
            
            #팩맨이 이동할 수 있는 state 에서 모든 유령들이 min 움직임을 했을 때 가장 큰 value 를 반환 
            v = max(v, min_value(newState, depth, 1))

        return v
    
    # min value를 반환하는 함수 => index != 0
    def min_value(state, depth, index):
        
        # 게임 종료 조건 확인 
        if state.isWin() or state.isLose():
            
            # 바로 현재 state 의 점수 반환 
            return state.getScore()
        
        # 깊이가 0 인 것은 더이상 미래의 상태를 탐색하지 x 
        if depth == 0:
            
            # 따라서, 현재 상태를 evaluation function 을 이용해서 평가 한 후, 그 값을 반환함. (게임 state 가 얼마나 유리한지 수치화)
            return self.evaluationFunction(state)
        
        # 현재 state 에서 index 차례(팩맨 또는 유령)일 때, 가능한 모든 action 들의 list 
        LegalActions = state.getLegalActions(index)
        
        # 현재, 가능한 action 이 없을 때, evaluation funciton 결과 반환 
        if len(LegalActions) == 0:
            return self.evaluationFunction(state)
        
        # value 를 infinity 로 initialize 
        v = float("inf")

        # 현재 index 가 마지막 agent 인지 확인 (팩맨이 첫번째로 움직이고 그 다음 유령들이 차례대로 움직이기 때문에 유령들이 모두 움직이고 나면 다음 턴으로 넘어가는 것.)
        if index == (state.getNumAgents() - 1):
            
            #각 가능한 행동들에 대해서 발생하는 successor state 를 평가하여, 그 중 min 결정 
            for LegalAction in LegalActions:
                
                # depth 를 넘어가면서 index 가 팩맨이 되기 때문에 max value function 사용 
                v = min(v, max_value(state.generateSuccessor(index, LegalAction), depth - 1))
        
        # 현재 index 가 마지막 agent 가 아닐 경우 
        else:
            for LegalAction in LegalActions:
                
                #index + 1 을 함으로써 depth 는 유지하고 min value 함수를 재귀적으로 호출 함. 
                v = min(v, min_value(state.generateSuccessor(index, LegalAction), depth, index + 1))
            
        return v
        
    #Returns the successor game state after an agent takes an action
    # 여기서 사용되는 action 은 가능한 모든 action들에서 각각 for loop 돌린 action들. 
    state = gameState.generateSuccessor(0, action)

    #위의 state 에서의 min_value 반환. 
    return min_value(state, self.depth, 1)
  
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER

    vmax = float("-inf")
    move = [Directions.STOP]
    # minimax 와 다른 점은 위에서는 self.index 인데 여기서는 pacmac 을 index 로 가짐. 
    # 유령이 random 하게 움직여도 팩맨은 최적의 경로를 찾아야함. 
    # 그래서 유령의 움직임에 대해 평균적인 결과를 계산하도록 함. 
    actions = gameState.getLegalActions(0)  
    for action in actions:
      v = self.getQ(gameState, action)
      if v > vmax:
        vmax = v
        move = [action]
      elif v == vmax:
        move.append(action)

    #print("expectimax", vmax)
    return random.choice(move)
  
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER

    # max_value function 은 minimax 와 동일 
    def max_value(state, depth):
      if state.isWin() or state.isLose():
        return state.getScore()
      if depth == 0:
        return self.evaluationFunction(state)
      
      LegalActions = state.getLegalActions(0)
      
      if len(LegalActions) == 0:
        return self.evaluationFunction(state)
      
      pacmanState = [state.generateSuccessor(0, LegalAction) for LegalAction in LegalActions]
      
      v = float("-inf")
      for newState in pacmanState:
        v = max(v, min_value(newState, depth, 1))
      return v
        
    def min_value(state, depth, index):
      if state.isWin() or state.isLose():
        return state.getScore()
      if depth == 0:
        return self.evaluationFunction(state)
      
      LegalActions = state.getLegalActions(index)
      
      if len(LegalActions) == 0:
        return self.evaluationFunction(state)
      
      # minimax 에서는 최악의 경우로 initialize 한 후 min 을 찾지만 
      # expecitimax 에서는 확률적인 결과의 기댓값을 계산함. 
      v = 0
      if index == state.getNumAgents() - 1:
        for LegalAction in LegalActions:
          # (팩맨) 가능한 모든 action 과 이에 따른 succesor state 에서의 max 값들의 평균값 저장 
          result = max_value(state.generateSuccessor(index, LegalAction), depth)
          if result is not None:
              v += result / len(LegalActions)
      else:
        for LegalAction in LegalActions:
          # (유령) 가능한 모든 action 과 이에 따른 succesor state 에서의 min 값들의 평균값 저장 
          result = min_value(state.generateSuccessor(index, LegalAction), depth, index + 1)
          if result is not None:
              v += result / len(LegalActions)
        
        return v

    # minimax 와 동일       
    state = gameState.generateSuccessor(0, action)
    return min_value(state, self.depth, 1)
  
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER

    # minimax 와 동일함 
    vmax = float("-inf")
    move = [Directions.STOP]
    actions = gameState.getLegalActions(self.index)
    for action in actions:
      v = self.getQ(gameState, action)
      if v > vmax:
        vmax = v
        move = [action]
      elif v == vmax:
        move.append(action)
    return random.choice(move)
  
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER

    # max_value function 은 minimax 와 동일 
    def max_value(state, depth):
      if state.isWin() or state.isLose():
        return state.getScore()
      if depth == 0:
        return self.evaluationFunction(state)
      
      LegalActions = state.getLegalActions(0)
      
      if len(LegalActions) == 0:
        return self.evaluationFunction(state)
      
      pacmanState = [state.generateSuccessor(0, LegalAction) for LegalAction in LegalActions]
      
      v = float("-inf")
      for newState in pacmanState:
        v = max(v, min_value(newState, depth, 1))
      return v
        
    def min_value(state, depth, index):
      if state.isWin() or state.isLose():
        return state.getScore()
      if depth == 0:
        return self.evaluationFunction(state)
      
      LegalActions = state.getLegalActions(index)
      
      if len(LegalActions) == 0:
        return self.evaluationFunction(state)
      
      # minimax 에서는 최악의 경우로 initialize 한 후 min 을 찾지만 
      # biased expecitimax 에서는 확률적인 결과의 기댓값을 계산함. 
      v = 0
      # 마지막 유령이 움직이고 이제 팩맨이 움직일 차례
      if index == state.getNumAgents() - 1: 
        for LegalAction in LegalActions:
          p = 0.5 / len(LegalActions)

          if LegalAction == Directions.STOP:
            p += 0.5

          #확률에 따른 max value 합산 결과 (팩맨)
          v += (max_value(state.generateSuccessor(index, LegalAction), depth - 1) * p)

      # 유령이 움직이는 차례 
      else:
        for LegalAction in LegalActions:
          p = 0.5 / len(LegalActions)

          if LegalAction == Directions.STOP:
            p += 0.5

          #확률에 따른 min value 합산 결과 (유령)
          result = min_value(state.generateSuccessor(index, LegalAction), depth, index + 1)
          if result is not None:
              v += result * p       
            
        return v

    # minimax 와 동일       
    state = gameState.generateSuccessor(0, action)
    return min_value(state, self.depth, 1)
  
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
  
    # minimax 와 동일함 
    vmax = float("-inf")
    move = [Directions.STOP]
    actions = gameState.getLegalActions(self.index)
    for action in actions:
      v = self.getQ(gameState, action)
      if v > vmax:
        vmax = v
        move = [action]
      elif v == vmax:
        move.append(action)

    #print("vmax: ", vmax)
    return random.choice(move)

    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    # max_value function 은 minimax 와 동일 
    def max_value(state, depth):
      if state.isWin() or state.isLose():
        return state.getScore()
      if depth == 0:
        return self.evaluationFunction(state)
      
      LegalActions = state.getLegalActions(0)
      
      if len(LegalActions) == 0:
        return self.evaluationFunction(state)
      
      pacmanState = [state.generateSuccessor(0, LegalAction) for LegalAction in LegalActions]
      
      v = float("-inf")
      for newState in pacmanState:
        v = max(v, min_value(newState, depth, 1))
      return v
        
    def min_value(state, depth, index):
      if state.isWin() or state.isLose():
        return state.getScore()
      if depth == 0:
        return self.evaluationFunction(state)
      
      LegalActions = state.getLegalActions(index)
      
      if len(LegalActions) == 0:
        return self.evaluationFunction(state)
      
      #여기서 부터 수정 계속하기 !!! 
      SuccStates = [state.generateSuccessor(index, LegalAction) for LegalAction in LegalActions]
      v = float("inf")

      # 마지막 유령이 움직이고 이제 팩맨이 움직일 차례
      if index == state.getNumAgents() - 1: 
        if (index % 2 == 1): #50% 확률로 minimax 
          for SuccState in SuccStates: 
            v = min(v, max_value(SuccState, depth - 1))
        else: #나머지 50%의 확률로 expectimax 
          v = 0
          result = 0
          for SuccState in SuccStates:
            result += max_value(SuccState, depth - 1)
          if result is not None:
            v = result / len(SuccStates)

      # 유령이 움직이는 차례 
      else:
        if (index % 2 == 1): #50% 확률로 minimax 
          for SuccState in SuccStates: 
            v = min(v, min_value(SuccState, depth, index + 1))
        else: #나머지 50%의 확률로 expectimax 
          v = 0
          result = 0
          for SuccState in SuccStates:
            result += min_value(SuccState, depth, index + 1)
          if result is not None: 
            v = result / len(SuccStates)      

      return v 
    
    # minimax 와 동일       
    state = gameState.generateSuccessor(0, action)

    return min_value(state, self.depth, 1)

    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER

    # minimax 와 동일함 
    vmax = float("-inf")
    move = [Directions.STOP]
    actions = gameState.getLegalActions(self.index)
    for action in actions:
      v = self.getQ(gameState, action)
      if v > vmax:
        vmax = v
        move = [action]
      elif v == vmax:
        move.append(action)

    #print("vmax: ", vmax)
    return random.choice(move)
  
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    # minimax max_value function에서 alpha 와 beta 가 추가됨  
    def max_value(state, depth, alpha, beta):
      if state.isWin() or state.isLose():
        return state.getScore()
      if depth == 0:
        return self.evaluationFunction(state)
      
      LegalActions = state.getLegalActions(0)
      
      if len(LegalActions) == 0:
        return self.evaluationFunction(state)
      
      pacmanState = [state.generateSuccessor(0, LegalAction) for LegalAction in LegalActions]
      
      v = float("-inf")
      for newState in pacmanState:
        v = max(v, min_value(newState, depth, 1, alpha, beta))
        alpha = max(alpha, v)
        if alpha >= beta:
          break
        
      return v
        
    #alpha beta 값을 input 으로 받아옴 
    def min_value(state, depth, index, alpha, beta):
      if state.isWin() or state.isLose():
        return state.getScore()
      if depth == 0:
        return self.evaluationFunction(state)
      
      LegalActions = state.getLegalActions(index)
      
      if len(LegalActions) == 0:
        return self.evaluationFunction(state)
      
      SuccStates = [state.generateSuccessor(index, LegalAction) for LegalAction in LegalActions]
      v = float("inf")

      # 마지막 유령이 움직이고 이제 팩맨이 움직일 차례
      if index == state.getNumAgents() - 1: 
        for SuccState in SuccStates: 
            v = min(v, max_value(SuccState, depth - 1, alpha, beta))
            beta = min(beta, v)
            if alpha >= beta: 
              break

      # 유령이 움직이는 차례 
      else:
          for SuccState in SuccStates:
            v = min(v, min_value(SuccState, depth, index + 1, alpha, beta))
            beta = min(beta, v)
            if alpha >= beta: 
              break   

      return v 
    
    # minimax 에서 initialized 된 alpha, beta 추가       
    state = gameState.generateSuccessor(0, action)
    alpha = float("-inf")
    beta = float("inf")

    return min_value(state, self.depth, 1, alpha, beta)
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
  
  # 현재 획득한 score 로 initialize 
  score = currentGameState.getScore()
    
  # 현재 팩맨의 위치 
  pacmanPosition = currentGameState.getPacmanPosition()
  
  # 첫번째 (유령과 팩맨 사이 거리)
  # 현재 유령들의 위치 
  ghostStates = currentGameState.getGhostStates()
  # 각 유령 별로 위치 list 로 저장 
  ghostPositions = [ghostState.getPosition() for ghostState in ghostStates]
  # -10 으로 가정 
  ghostDistanceWeight = -1.5
  # manhattan distance 를 사용해서 팩맨과 각 유령 사이의 거리 계산 
  expectedGhostDistances = [util.manhattanDistance(pacmanPosition, ghostPos) for ghostPos in ghostPositions]
  
  for dist in expectedGhostDistances:
      # distance 가 1 보다 클 때, 즉 Pac-Man과 유령이 겹치지 않을 때
      if dist > 0:  
          # 점수 업데이트 하기 (더해질때마다 감소됨)
          score += ghostDistanceWeight / dist
  
  # 두번째 (남은 음식과 팩맨 사이 거리 )
  # 현재 상태에서 남은 음식의 위치 list 로 저장 
  foodList = currentGameState.getFood().asList()
  foodDistanceWeight = -1.1 # 더 적은 음식이 있는 상태를 선호하도록 설정
  
  # 팩맨과 남은 음식과의 거리 계산 (mahattan distance)
  foodDistances = [util.manhattanDistance(pacmanPosition, foodPos) for foodPos in foodList]
  # 남은 음식이 1개 이상 존재 할 때, 
  if foodDistances:
      # 남은 음식과 팩맨과의 평균 거리 계산
      avgFoodDistance = sum(foodDistances) / len(foodDistances)
      # 점수 업데이트 하기 (더해질때마다 감소)
      score += foodDistanceWeight * avgFoodDistance

  # 세번째 (캡슐 -- 유령 하얗게 만드는거)
  # 현재 남아있는 모든 캡슐의 위치 가져오기 
  capsulePositions = currentGameState.getCapsules()
  capsulesCountWeight = -29 #29
  
  # 남은 캡슐의 개수를 점수에 반영
  score += capsulesCountWeight * len(capsulePositions)
  
  return score

  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER

  #위의 agent 들 중 alphabeta agent 의 값이 가장 크게 나와 alphabeta 로 사용. 
  return 'AlphaBetaAgent'

  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
