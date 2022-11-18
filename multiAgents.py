# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        successorGameStateScore = successorGameState.getScore()
        value = successorGameState.getScore()
        if not successorGameState.isLose() and not successorGameState.isWin():
            _, distanceToClosestDot = findClosestDot(newFood.asList(), newPosition)
            value += 1/distanceToClosestDot
        return value

def findClosestDot(dots: tuple, currentPosition):
    """
    Utility function for finding closest dot used by evaluationFunction of ReflexAgent
    """
    positionOfClosestDot = None
    distanceToClosestDot = None
    for dot in dots:
        distance = manhattanDistance(currentPosition, dot)
        
        if positionOfClosestDot == None or distance < distanceToClosestDot:
            distanceToClosestDot = distance
            positionOfClosestDot = dot

    return positionOfClosestDot, distanceToClosestDot

def scoreEvaluationFunction(currentGameState: GameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        _, optimalAction = self.minimaxDecision(gameState, self.depth, 0)
        return optimalAction

    def reachedTerminalState(self, gameState, depth):
        return gameState.isWin() or gameState.isLose() or depth == 0

    def minimaxDecision(self, gameState, depth, agent):
        return self.findMax(gameState, agent, depth) if agent == 0 else self.findMin(gameState, agent, depth)

    def findMax(self, gameState, agent, depth):
        if self.reachedTerminalState(gameState, depth):
            return self.evaluationFunction(gameState), Directions.STOP

        maxUtilityValue = None
        for action in gameState.getLegalActions(agent):
            successorGameState = gameState.generateSuccessor(agent, action)

            utilityValue, _ = \
            self.minimaxDecision(successorGameState, depth - 1, 0) \
            if agent == gameState.getNumAgents() - 1 \
            else \
            self.minimaxDecision(successorGameState, depth, agent + 1)

            if maxUtilityValue == None or utilityValue > maxUtilityValue:
                maxAction = action
                maxUtilityValue = utilityValue

        return maxUtilityValue, maxAction

    def findMin(self, gameState, agent, depth):
        if self.reachedTerminalState(gameState, depth):
            return self.evaluationFunction(gameState), Directions.STOP

        minUtilityValue = None
        for action in gameState.getLegalActions(agent):
            successorGameState = gameState.generateSuccessor(agent, action)

            utilityValue, _ = \
            self.minimaxDecision(successorGameState, depth - 1, 0) \
            if agent == gameState.getNumAgents() - 1 \
            else \
            self.minimaxDecision(successorGameState, depth, agent + 1)

            if minUtilityValue == None or utilityValue < minUtilityValue:
                minAction = action
                minUtilityValue = utilityValue

        return minUtilityValue, minAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        _, optimalAction = self.minimaxDecision(gameState, self.depth, 0, None, None);
        return optimalAction

    def reachedTerminalState(self, gameState, depth):
        return gameState.isWin() or gameState.isLose() or depth == 0

    def minimaxDecision(self, gameState, depth, agent, alpha, beta):
        return self.findMax(gameState, agent, depth, alpha, beta) if agent == 0 else self.findMin(gameState, agent, depth, alpha, beta)

    def findMax(self, gameState, agent, depth, alpha, beta):
        if self.reachedTerminalState(gameState, depth):
            return self.evaluationFunction(gameState), Directions.STOP

        maxUtilityValue = None
        for action in gameState.getLegalActions(agent):
            successorGameState = gameState.generateSuccessor(agent, action)

            utilityValue, _ = \
            self.minimaxDecision(successorGameState, depth - 1, 0, alpha, beta) \
            if agent == gameState.getNumAgents() - 1 \
            else \
            self.minimaxDecision(successorGameState, depth, agent + 1, alpha, beta)

            if maxUtilityValue == None or utilityValue > maxUtilityValue:
                maxAction = action
                maxUtilityValue = utilityValue

            if beta != None and maxUtilityValue > beta:
                break

            if alpha == None or maxUtilityValue > alpha:
                alpha = maxUtilityValue

        return maxUtilityValue, maxAction

    def findMin(self, gameState, agent, depth, alpha, beta):
        if self.reachedTerminalState(gameState, depth):
            return self.evaluationFunction(gameState), Directions.STOP

        minUtilityValue = None
        for action in gameState.getLegalActions(agent):
            successorGameState = gameState.generateSuccessor(agent, action)

            utilityValue, _ = \
            self.minimaxDecision(successorGameState, depth - 1, 0, alpha, beta) \
            if agent == gameState.getNumAgents() - 1 \
            else \
            self.minimaxDecision(successorGameState, depth, agent + 1, alpha, beta)

            if minUtilityValue == None or utilityValue < minUtilityValue:
                minAction = action
                minUtilityValue = utilityValue

            if alpha != None and minUtilityValue < alpha:
                break

            if beta == None or minUtilityValue < beta:
                beta = minUtilityValue

        return minUtilityValue, minAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        _, action = self.expectimaxDecision(gameState, depth = self.depth, agent = 0)
        return action

    def expectimaxDecision(self, gameState, depth, agent):
        return self.findMax(gameState, agent, depth) if agent == 0 else self.findChance(gameState, agent, depth)

    def reachedTerminalState(self, gameState, depth):
        return gameState.isWin() or gameState.isLose() or depth == 0

    def findMax(self, gameState, agent, depth):
        if self.reachedTerminalState(gameState, depth):
            return self.evaluationFunction(gameState), Directions.STOP

        maxUtilityValue = None
        for action in gameState.getLegalActions(agent):
            successorGameState = gameState.generateSuccessor(agent, action)

            utilityValue, _ = \
            self.expectimaxDecision(successorGameState, depth - 1, 0) \
            if agent == gameState.getNumAgents() - 1 \
            else \
            self.expectimaxDecision(successorGameState, depth, agent + 1)

            if maxUtilityValue == None or utilityValue > maxUtilityValue:
                maxAction = action
                maxUtilityValue = utilityValue

        return maxUtilityValue, maxAction

    def findMin(self, gameState, agent, depth):
        if self.reachedTerminalState(gameState, depth):
            return self.evaluationFunction(gameState), Directions.STOP

        minUtilityValue = None
        for action in gameState.getLegalActions(agent):
            successorGameState = gameState.generateSuccessor(agent, action)

            utilityValue, _ = \
            self.expectimaxDecision(successorGameState, depth - 1, 0) \
            if agent == gameState.getNumAgents() - 1 \
            else \
            self.expectimaxDecision(successorGameState, depth, agent + 1)

            if minUtilityValue == None or utilityValue < minUtilityValue:
                minAction = action
                minUtilityValue  = utilityValue

        return minUtilityValue, minAction


    def findChance(self, gameState, agent, depth):
        if self.reachedTerminalState(gameState, depth):
            return self.evaluationFunction(gameState), Directions.STOP

        utilityValue = 0
        legalActionsOfAgent = gameState.getLegalActions(agent)
        for action in legalActionsOfAgent:
            successorGameState = gameState.generateSuccessor(agent, action)

            utilityValue += \
            (1/len(legalActionsOfAgent)) * (self.expectimaxDecision(successorGameState, depth - 1, 0))[0] \
            if agent == gameState.getNumAgents() - 1 \
            else \
            (1/len(legalActionsOfAgent)) * (self.expectimaxDecision(successorGameState, depth, agent + 1))[0]

            randomMove = random.choice(legalActionsOfAgent)

        return utilityValue, randomMove

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    value = currentGameState.getScore()
    if not currentGameState.isLose() and not currentGameState.isWin():
        value = currentGameState.getScore()

        _, distanceToClosestDot = findClosestDot(currentGameState.getFood().asList(), currentGameState.getPacmanPosition())
        _, distanceToClosestGhost = findClosestDot(currentGameState.getGhostPositions(), currentGameState.getPacmanPosition())

        value += 1/distanceToClosestDot - distanceToClosestGhost
        if(currentGameState.getCapsules()):
            _, distanceToClosestCapsule = findClosestDot(currentGameState.getCapsules(), currentGameState.getPacmanPosition())
            value += 1/distanceToClosestCapsule
    return value

# Abbreviation
better = betterEvaluationFunction
