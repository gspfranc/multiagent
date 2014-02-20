# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = min(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        def distance(a, b):
            x1, y1 = a
            x2, y2 = b
            return abs(x1 - x2) + abs(y1 - y2)

        closestFood = min(map(lambda food: distance(food, newPos), newFood.asList())) if len(
            newFood.asList()) != 0 else 0
        remainingFoodScore = -closestFood - 1 if len(currentGameState.getFood().asList()) > len(
            successorGameState.getFood().asList()) else 0
        ghostScore = 0
        for ghost in newGhostStates:
            if distance(ghost.configuration.pos, newPos) <= 1:
                ghostScore += 100000

        return closestFood + remainingFoodScore + ghostScore


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def maxValue(self, gameState, depth):
        v = -sys.maxint, None
        actions = gameState.getLegalActions(0)
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState), None
        depth += 1
        for action in actions:
            nv = self.minValue(gameState.generateSuccessor(0, action), depth)[0], action
            v = nv if nv[0] > v[0] else v
        return v

    def minValue(self, gameState, depth, agent = 1):
        v = sys.maxint, None
        actions = gameState.getLegalActions(agent)
        if len(actions) == 0:
            return self.evaluationFunction(gameState), None
        for action in gameState.getLegalActions(agent):
            if agent + 1 == gameState.getNumAgents():
                nv = self.maxValue(gameState.generateSuccessor(agent, action), depth)[0], action
            else:
                nv = self.minValue(gameState.generateSuccessor(agent, action), depth, agent + 1)[0], action
            v = nv if nv[0] < v[0] else v
        return v


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return self.maxValue(gameState, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, gameState, depth, alpha, beta):
        v = -sys.maxint, None
        actions = gameState.getLegalActions(0)
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState), None
        depth += 1
        for action in actions:
            nv = self.minValue(gameState.generateSuccessor(0, action), depth, alpha, beta)[0], action
            v = nv if nv[0] > v[0] else v
            if v[0] > beta:
                return v
            alpha = max(alpha, v[0])
        return v

    def minValue(self, gameState, depth, alpha, beta, agent = 1):
        v = sys.maxint, None
        actions = gameState.getLegalActions(agent)
        if len(actions) == 0:
            return self.evaluationFunction(gameState), None
        for action in gameState.getLegalActions(agent):
            if agent + 1 == gameState.getNumAgents():
                nv = self.maxValue(gameState.generateSuccessor(agent, action), depth, alpha, beta)[0], action
            else:
                nv = self.minValue(gameState.generateSuccessor(agent, action), depth, alpha, beta, agent + 1)[0], action
            v = nv if nv[0] < v[0] else v
            if v[0] < alpha:
                return v
            beta = min(beta, v[0])
        return v


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return self.maxValue(gameState, 0, -sys.maxint, sys.maxint)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        def expectedvalue(gameState, agentindex, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            numghosts = gameState.getNumAgents() - 1
            actions = gameState.getLegalActions(agentindex)
            total = 0
            for action in actions:
                next_state = gameState.generateSuccessor(agentindex, action)
                if agentindex == numghosts:
                    total += maxvalue(next_state, depth + 1)
                else:
                    total += expectedvalue(next_state, agentindex + 1, depth)
            return total / len(actions)

        def maxvalue(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(0)
            score = -sys.maxint
            for action in actions:
                next_state = gameState.generateSuccessor(0, action)
                score = max(score, expectedvalue(next_state, 1, depth))
            return score

        actions = gameState.getLegalActions(0)
        next_states = [gameState.generateSuccessor(0, action) for action in actions]
        scores = [expectedvalue(state, 1, 0) for state in next_states]
        best_action_index = scores.index(max(scores))
        return actions[best_action_index]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Base score on amount of food left and distance to closest food.
      Sometime pacman got stuck behind a wall so we added a random factor.
    """
    if currentGameState.isWin():
        return sys.maxint
    if currentGameState.isLose():
        return -sys.maxint

    pacman = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    capsule = currentGameState.getCapsules()

    def distance(a, b):
        x1, y1 = a
        x2, y2 = b
        return abs(x1 - x2) + abs(y1 - y2)

    closest_food = min(distance(pacman, x) for x in food)
    food_count = len(food)
    food_score = - closest_food - (food_count * 100)

    ghost_score = 0
    for ghost in ghosts:
        if distance(ghost.configuration.pos, pacman) <= 1 and ghost.scaredTimer < 3:
            ghost_score -= 1000

    capsule_score = -len(capsule) * 3

    return scoreEvaluationFunction(currentGameState) + food_score + ghost_score + capsule_score + random.randint(0, 3)

# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

