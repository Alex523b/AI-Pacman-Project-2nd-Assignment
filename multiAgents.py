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
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        value = successorGameState.getScore()
        if not successorGameState.isLose() and not successorGameState.isWin():
            _, distanceToClosestDot = findClosestEntity(newFood.asList(), newPosition)
            value += 1/distanceToClosestDot
        return value

def findClosestEntity(coordinatesOfEntities: tuple, currentPosition):
    """
    This utility function returns the position-coordinates of the closest "entity" and its distance from the current position.
    "entity" can be a ghost, food, or capsule, since they are represented by coordinates in 2D.
    """
    positionOfClosestEntity = None
    distanceToClosestEntity = None
    for coordinatesOfEntity in coordinatesOfEntities:
        distance = manhattanDistance(currentPosition, coordinatesOfEntity)
        
        if positionOfClosestEntity == None or distance < distanceToClosestEntity:
            distanceToClosestEntity = distance
            positionOfClosestEntity = coordinatesOfEntity

    return positionOfClosestEntity, distanceToClosestEntity

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
        _, optimalAction = self.minimaxDecision(gameState, 0, self.depth)
        return optimalAction

    def reachedTerminalState(self, gameState, depth):
        return gameState.isWin() or gameState.isLose() or depth == 0

    def minimaxDecision(self, gameState, agent, depth):
        """
        All agents make optimal decisions; call the appropriate helper functions
        """
        return self.findMax(gameState, agent, depth) if agent == 0 else self.findMin(gameState, agent, depth)

    def findMax(self, gameState, agent, depth):
        if self.reachedTerminalState(gameState, depth):
            return self.evaluationFunction(gameState), Directions.STOP

        maxUtilityValue = None
        for action in gameState.getLegalActions(agent):
            successorGameState = gameState.generateSuccessor(agent, action)

            utilityValue, _ = \
            self.minimaxDecision(successorGameState, 0, depth - 1) \
            if agent == gameState.getNumAgents() - 1 \
            else \
            self.minimaxDecision(successorGameState, agent + 1, depth)

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
            self.minimaxDecision(successorGameState, 0, depth - 1) \
            if agent == gameState.getNumAgents() - 1 \
            else \
            self.minimaxDecision(successorGameState, agent + 1, depth)

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
        _, optimalAction = self.minimaxDecision(gameState, 0, self.depth, None, None);
        return optimalAction

    def reachedTerminalState(self, gameState, depth):
        return gameState.isWin() or gameState.isLose() or depth == 0

    def minimaxDecision(self, gameState, agent, depth, alpha, beta):
        return self.findMax(gameState, agent, depth, alpha, beta) if agent == 0 else self.findMin(gameState, agent, depth, alpha, beta)

    def findMax(self, gameState, agent, depth, alpha, beta):
        if self.reachedTerminalState(gameState, depth):
            return self.evaluationFunction(gameState), Directions.STOP

        maxUtilityValue = None
        for action in gameState.getLegalActions(agent):
            successorGameState = gameState.generateSuccessor(agent, action)

            utilityValue, _ = \
            self.minimaxDecision(successorGameState, 0, depth - 1, alpha, beta) \
            if agent == gameState.getNumAgents() - 1 \
            else \
            self.minimaxDecision(successorGameState, agent + 1, depth, alpha, beta)

            if maxUtilityValue == None or utilityValue > maxUtilityValue:
                maxAction = action
                maxUtilityValue = utilityValue

            if beta != None and maxUtilityValue > beta: # Prune condition
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
            self.minimaxDecision(successorGameState, 0, depth - 1, alpha, beta) \
            if agent == gameState.getNumAgents() - 1 \
            else \
            self.minimaxDecision(successorGameState, agent + 1, depth, alpha, beta)

            if minUtilityValue == None or utilityValue < minUtilityValue:
                minAction = action
                minUtilityValue = utilityValue

            if alpha != None and minUtilityValue < alpha: # Prune condition
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
        _, action = self.expectimaxDecision(gameState, 0, self.depth)
        return action

    def expectimaxDecision(self, gameState, agent, depth):
        """
        Pac-Man makes optimal decisions (thus, call findMax()), while ghosts don't (hence, call findChance()).
        """
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
            self.expectimaxDecision(successorGameState, 0, depth - 1) \
            if agent == gameState.getNumAgents() - 1 \
            else \
            self.expectimaxDecision(successorGameState, agent + 1, depth)

            if maxUtilityValue == None or utilityValue > maxUtilityValue:
                maxAction = action
                maxUtilityValue = utilityValue

        return maxUtilityValue, maxAction

    def findChance(self, gameState, agent, depth):
        if self.reachedTerminalState(gameState, depth):
            return self.evaluationFunction(gameState), Directions.STOP

        utilityValue = 0
        legalActionsOfAgent = gameState.getLegalActions(agent)
        numberOfLegalActions = len(legalActionsOfAgent)
        for action in legalActionsOfAgent:
            successorGameState = gameState.generateSuccessor(agent, action)

            """
            An adversary chooses amongst their legal actions uniformly at random
            """
            utilityValue += \
            (1/numberOfLegalActions) * (self.expectimaxDecision(successorGameState, 0, depth - 1))[0] \
            if agent == gameState.getNumAgents() - 1 \
            else \
            (1/numberOfLegalActions) * (self.expectimaxDecision(successorGameState, agent + 1, depth))[0]

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

        _, distanceToClosestDot = findClosestEntity(currentGameState.getFood().asList(), currentGameState.getPacmanPosition())
        _, distanceToClosestGhost = findClosestEntity(currentGameState.getGhostPositions(), currentGameState.getPacmanPosition())

        value += 1/distanceToClosestDot - distanceToClosestGhost
        if(currentGameState.getCapsules()):
            _, distanceToClosestCapsule = findClosestEntity(currentGameState.getCapsules(), currentGameState.getPacmanPosition())
            value += 1/distanceToClosestCapsule

    return value

# Abbreviation
better = betterEvaluationFunction
