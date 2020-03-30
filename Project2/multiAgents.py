###
#   THIS CODE WAS MY OWN WORK , IT WAS WRITTEN WITHOUT CONSULTING ANY
#    SOURCES OUTSIDE OF THOSE APPROVED BY THE INSTRUCTOR. Matthew Chau
###

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
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

import random

import util
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
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()  # food available from successor state (excludes food@successor)
        newFoodList = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()

        # Avoid the pacman stopping for no reason
        if action == 'Stop':
            return -float('inf')

        # Find the closest food
        minFoodDistance = float('inf')
        for foodPos in newFoodList:
            minFoodDistance = min(minFoodDistance, util.manhattanDistance(foodPos, newPos))

        # If the ghost is near, do not apply that action
        for ghostState in newGhostStates:
            if util.manhattanDistance(ghostState.getPosition(), newPos) < 2 and ghostState.scaredTimer == 0:
                return -float('inf')

        return successorGameState.getScore() + 1.0 / minFoodDistance


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
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

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

        # Returns the action with the maximum action value
        return self.maxValue(gameState, 0, 0)[0]

    def minimax(self, gameState, agentIndex, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)[1]
        if agentIndex >= 1:
            return self.minValue(gameState, agentIndex, depth)[1]

    # Returns a (action, action value) tuple with the maximum value for the max player (pacman)
    def maxValue(self, gameState, agentIndex, depth):
        maxEval = ("action", -float('inf'))
        for action in gameState.getLegalActions(agentIndex):
            maxEval = max(maxEval, (action, self.minimax(gameState.generateSuccessor(agentIndex, action),
                                                         (depth + 1) % gameState.getNumAgents(), depth + 1)),
                          key=lambda x: x[1])
        return maxEval

    # Returns a (action, action value) tuple with the minimum value for the min player (ghosts)
    def minValue(self, gameState, agentIndex, depth):
        minEval = ("action", float('inf'))
        for action in gameState.getLegalActions(agentIndex):
            minEval = min(minEval, (action, self.minimax(gameState.generateSuccessor(agentIndex, action),
                                                         (depth + 1) % gameState.getNumAgents(), depth + 1)),
                          key=lambda x: x[1])
        return minEval


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.maxValue(gameState, 0, 0, -float('inf'), float('inf'))[0]

    def minimax(self, gameState, agentIndex, depth, alpha, beta):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)[1]
        if agentIndex >= 1:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)[1]

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        maxEval = ("action", -float('inf'))
        for action in gameState.getLegalActions(agentIndex):
            maxEval = max(maxEval, (action, self.minimax(gameState.generateSuccessor(agentIndex, action),
                                                         (depth + 1) % gameState.getNumAgents(), depth + 1, alpha,
                                                         beta)),
                          key=lambda x: x[1])
            # Avoid unnecessary computation
            if maxEval[1] > beta:
                return maxEval
            alpha = max(alpha, maxEval[1])
        return maxEval

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        minEval = ("action", float('inf'))
        for action in gameState.getLegalActions(agentIndex):
            minEval = min(minEval, (action, self.minimax(gameState.generateSuccessor(agentIndex, action),
                                                         (depth + 1) % gameState.getNumAgents(), depth + 1, alpha,
                                                         beta)),
                          key=lambda x: x[1])
            # Avoid unnecessary computation
            if minEval[1] < alpha:
                return minEval
            beta = min(beta, minEval[1])
        return minEval


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

        # Returns the action with the maximum expected value
        return self.maxValue(gameState, 0, 0)[0]

    def expectedMinimax(self, gameState, agentIndex, depth):
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)[1]
        if agentIndex >= 1:
            return self.expectedValue(gameState, agentIndex, depth)[1]

    def maxValue(self, gameState, agentIndex, depth):
        maxEval = ("action", -float('inf'))
        for action in gameState.getLegalActions(agentIndex):
            maxEval = max(maxEval, (action, self.expectedMinimax(gameState.generateSuccessor(agentIndex, action),
                                                                 (depth + 1) % gameState.getNumAgents(), depth + 1)),
                          key=lambda x: x[1])
        return maxEval

    # Returns a (action, expected value) where expected value = sum of (prob * evaluation of each successor state)
    def expectedValue(self, gameState, agentIndex, depth):
        expectedEval = ("action", 0)
        possibleActions = gameState.getLegalActions(agentIndex)
        prob = 1.0 / len(possibleActions)
        for action in possibleActions:
            expectedEval = action, expectedEval[1] + prob * self.expectedMinimax(
                gameState.generateSuccessor(agentIndex, action), (depth + 1) % gameState.getNumAgents(), depth + 1)
        return expectedEval


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    foodPos = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capPos = currentGameState.getCapsules()
    currentPos = currentGameState.getPacmanPosition()

    # The less food left, the higher the score
    foodNum = len(foodPos)

    # The farther away from ghosts, the higher the score
    ghostDist = [util.manhattanDistance(ghostState.getPosition(), currentPos) for ghostState in ghostStates]

    # The closer the closest food is, the higher the score
    minFoodDist = min([util.manhattanDistance(food, currentPos) for food in foodPos]) if foodPos else -float('inf')

    # The less capsule left, the higher the score
    capNum = len(capPos)

    foodNumCoeff = 1000000
    minFoodDistCoeff = 500
    capNumCoeff = 10000

    # Close to ghosts
    if ghostDist[0] < 1 or ghostDist[1] < 1:
        return -99999999999

    return sum(ghostDist) + foodNumCoeff / (foodNum + 1) + minFoodDistCoeff / (minFoodDist + 1) + capNumCoeff / (
                capNum + 1)


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
        util.raiseNotDefined()
