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


import random

import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


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
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        x_cord = newPos[0]
        y_cord = newPos[1]
        curr_food = currentGameState.getFood()

        # Plan: 3 components
        # 1: give eval to food
        # 2: the further away to ghost the better
        # 3: blackout when too close to ghost

        eval = curr_food[x_cord][y_cord]

        # print(newFood.height, newFood.width)

        # eval -= ghost_too_close_technique(newGhostStates, newPos, 1, 30)
        eval -= ghost_too_close_technique(newGhostStates, newPos, 1, 90)
        # eval -= ghost_scan_technique(newGhostStates, newPos)
        eval += food_scan_techique(curr_food, newPos)

        return int(eval)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

def find2DDistance(tuple1, tuple2):
    return abs(tuple1[0] - tuple2[0]) + abs(tuple1[1] - tuple2[1])

def ghost_too_close_technique(newGhostStates, position, threshold=1, reduction=10):
    """
    Lowers the return/eval value dramatically when the ghost is closer or equal
    to the threshold distance

    :param newGhostState:
    :param position:
    :return:
    """

    for ghost_state in newGhostStates:
        pos = ghost_state.getPosition()
        if find2DDistance(pos, position) <= threshold:
            return abs(reduction)

    return 0

def food_scan_techique(food_map, pos):
    """

    food scanning techique, used currently to increase reflex-agent's performance

    :param food_map: GameSate.getFood() (grid object)
    :param pos: current position (of pacman)
    :return: float value representing the food situation in respect
    to food_map and pos
    """

    # print(food_map.width, food_map.height)
    # print("test:")
    eval = 0
    for i in range(food_map.width):
        for j in range(food_map.height):
            if food_map[i][j]:
                food_distance = find2DDistance((i,j), (pos))
                if food_distance == 0:
                    food_distance = 0.5
                eval += 1/food_distance * 7

    # return food_map[food_map.width - 1]
    return eval


def ghost_scan_technique(ghost_states, position):
    """

    :param ghost_states: list that include all ghost states
    :param position: pacman's position
    :return: a float representing the ghost situation in respect to inputs
    """

    # The structure will be similar to what's seen in ghost_too_close_technique

    return_val = 0

    for ghost_state in ghost_states:
        pos = ghost_state.getPosition()
        ghost_distance = find2DDistance(pos, position)
        if ghost_distance == 0:
            return 10

        return_val += 1/ghost_distance

    return return_val


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)

      Minimax action from CURRENT gameState USING:
      self.depth, self.evaluationFunction

      gameState.getLegalActions(agentIndex):
      gameState.generateSuccessor(agentIndex, action):
      gameState.getNumAgents():
    """

    def getAction(self, gameState):

        def get_sucessors_states(agent_index, state):
            ret = []
            for action in state.getLegalActions(agent_index):
                ret.append((state.generateSuccessor(agent_index, action), action))
            return ret

        def min_move(remaining_depth, agent_index, state):
            successor_states = get_sucessors_states(agent_index, state)

            # The case where there exist no successors states
            if len(successor_states) == 0:
                return self.evaluationFunction(state), None

            # current_val represent (positive) inf atm
            current_val = None
            current_preferred_action = None

            for successor, action in successor_states:
                tentative_value, tentative_action = minimax_switcher(remaining_depth, agent_index+1, successor)
                # See if this new value is the new "min", while taking advantage of Python's lazy evaluation.
                if current_val is None or tentative_value < current_val:
                    # Replace if this is indeed the case, and as well as the action associated with it
                    current_val = tentative_value
                    current_preferred_action = action

            # Return the preferred action, and it's value
            return current_val, current_preferred_action
            

        def max_move(remaining_depth, agent_index, state):
            # Good to keep in mind that max_move will ask for a depth reduction        
            # For comments/documentations for steps within this function, refer
            # to min_move() that is above.
            successor_states = get_sucessors_states(agent_index, state)

            if len(successor_states) == 0:
                return self.evaluationFunction(state), None

            current_val = None
            current_preferred_action = None

            for successor, action in successor_states:
                tentative_value, tentative_action = minimax_switcher(remaining_depth-1, agent_index+1, successor)
                if current_val is None or tentative_value > current_val:
                    current_val = tentative_value
                    current_preferred_action = action

            return current_val, current_preferred_action

 
        def minimax_switcher(remaining_depth, agent_index, state):
            agent_index = agent_index % state.getNumAgents()

            # The only time remaining_depth becomes 0 is when last min finsihed
            # it's move, Pacman's move now
            if remaining_depth == 0 and agent_index == 0:
                # None is just a place holder
                return self.evaluationFunction(state), None
            if agent_index == 0:
                return max_move(remaining_depth, agent_index, state)
            if agent_index < state.getNumAgents():
                return min_move(remaining_depth, agent_index, state)
            else:
                raise Exception 
   
        minimax_score, action = minimax_switcher(self.depth, 0, gameState)

        # print(minimax_score, action)

        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)

      Minimax action from CURRENT gameState USING:
      self.depth, self.evaluationFunction

      gameState.getLegalActions(agentIndex):
      gameState.generateSuccessor(agentIndex, action):
      gameState.getNumAgents():
    """

    def getAction(self, gameState):
        """ 
            All code below based on q3's minimax agent.
            Structure remains all same.

            "Alpha is the best option for the max along the route
            Beta is the best option for min"
        """

        def min_move(remaining_depth, agent_index, state, alpha, beta):
            successor_actions = state.getLegalActions(agent_index)

            # Same as before, the case where there exist no successors states
            if len(successor_actions) == 0:
                return self.evaluationFunction(state), None

            current_val = None
            current_preferred_action = None

            for action in successor_actions:
                # We choose whether to expand on states or not here instead
                # TODO: Logic to expand states via alpha, beta
                # The first node of each min/max move must be explored:
                tentative_state = state.generateSuccessor(agent_index, action)

                tentative_value, tentative_action = alphabeta_switcher(remaining_depth, agent_index+1, tentative_state, alpha, beta)
                
                if tentative_value <= alpha:
                    # We can return now, knowing that max is never going to pick this state
                    return tentative_value, action
                if current_val is None or tentative_value < current_val:
                    # New current_val is assigned though tentative_value
                    current_val = tentative_value
                    current_preferred_action = action
                # Beta needs to be updated as well, if a lower score option is found.
                # TODO: Check embeddness
                if beta > current_val:
                    beta = current_val

            return current_val, current_preferred_action
            

        def max_move(remaining_depth, agent_index, state, alpha, beta):
            successor_actions = state.getLegalActions(agent_index)
            # Good to keep in mind that max_move will ask for a depth reduction
            # For comments/documentations for steps within this function, refer
            # to min_move() that is above.
            if len(successor_actions) == 0:
                return self.evaluationFunction(state), None

            current_val = None
            current_preferred_action = None

            for action in successor_actions:
                tentative_state = state.generateSuccessor(agent_index, action)
                tentative_value, tentative_action = alphabeta_switcher(remaining_depth-1, agent_index+1, tentative_state, alpha, beta)
                if tentative_value >= beta:
                    return tentative_value, action
                if current_val is None or tentative_value > current_val:
                    current_val = tentative_value
                    current_preferred_action = action
                if alpha < current_val:
                    alpha = current_val

            return current_val, current_preferred_action

 
        def alphabeta_switcher(remaining_depth, agent_index, state, alpha, beta):
            agent_index = agent_index % state.getNumAgents()

            # The only time remaining_depth becomes 0 is when last min finsihed
            # it's move, Pacman's move now
            if remaining_depth == 0 and agent_index == 0:
                # None is just a place holder
                return self.evaluationFunction(state), None
            if agent_index == 0:
                return max_move(remaining_depth, agent_index, state, alpha, beta)
            if agent_index < state.getNumAgents():
                return min_move(remaining_depth, agent_index, state, alpha, beta)
            else:
                # Should never reach here
                raise Exception 
   
        alphabeta_score, action = alphabeta_switcher(self.depth, 0, gameState, -999999, 999999)

        # print(minimax_score, action)

        return action


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

        def get_sucessors_states(agent_index, state):
            ret = []
            for action in state.getLegalActions(agent_index):
                ret.append((state.generateSuccessor(agent_index, action), action))
            return ret

        def min_move(remaining_depth, agent_index, state):
            successor_states = get_sucessors_states(agent_index, state)

            if len(successor_states) == 0:
                return self.evaluationFunction(state), None

            current_sum = 0

            for successor, action in successor_states:
                tentative_value, tentative_action = minimax_switcher(remaining_depth, agent_index+1, successor)
                current_sum += tentative_value

            return current_sum / len(successor_states), None


        def max_move(remaining_depth, agent_index, state):
            # Good to keep in mind that max_move will ask for a depth reduction
            # For comments/documentations for steps within this function, refer
            # to min_move() that is above.
            successor_states = get_sucessors_states(agent_index, state)

            if len(successor_states) == 0:
                return self.evaluationFunction(state), None

            current_val = None
            current_preferred_action = None

            for successor, action in successor_states:
                tentative_value, tentative_action = minimax_switcher(remaining_depth-1, agent_index+1, successor)
                if current_val is None or tentative_value > current_val:
                    current_val = tentative_value
                    current_preferred_action = action

            return current_val, current_preferred_action


        def minimax_switcher(remaining_depth, agent_index, state):
            agent_index = agent_index % state.getNumAgents()

            # The only time remaining_depth becomes 0 is when last min finsihed
            # it's move, Pacman's move now
            if remaining_depth == 0 and agent_index == 0:
                # None is just a place holder
                return self.evaluationFunction(state), None
            if agent_index == 0:
                return max_move(remaining_depth, agent_index, state)
            if agent_index < state.getNumAgents():
                return min_move(remaining_depth, agent_index, state)
            else:
                raise Exception

        minimax_score, action = minimax_switcher(self.depth, 0, gameState)

        # print(minimax_score, action)

        return action

def food_printer(food_grid):
    """Just a simple printer for the food gird
    aim to be cleaner than the default grid.__str__()
    """

    for y in range(food_grid.height):
        this_line = ""
        for x in range(food_grid.width):
            if food_grid[x][y]:
                this_line += "*"
            else:
                this_line += " "

        print(this_line)

def prevent_dead_corners(pos, currentGameState, curr_val, bad_state_val=-float("inf")):
    """This is to prevent pacman from going into the corners that
    provides no progress

    :returns float;     bad_state_val = -1 iff enters a dead corner
                        curr_val (unchanged utility) otherwise
    """

    x = pos[0]
    y = pos[1]

    # print(x,y)
    # print(currentGameState.getFood().height)

    walls_facing = int(currentGameState.getWalls()[x-1][y]) + \
                   int(currentGameState.getWalls()[x+1][y]) + \
                   int(currentGameState.getWalls()[x][y-1]) + \
                   int(currentGameState.getWalls()[x][y+1])

    if walls_facing >= 3:
        return bad_state_val

    return curr_val

def food_eval(currentGameState, current_val):
    """
    :param currentGameState: currentGameState
    :param current_val: current_utility
    :return: modified current_ulitity according to current_val and currentGameState
    """

    food_map = currentGameState.getFood()
    pos = currentGameState.getPacmanPosition()
    # food_remaining = currentGameState.getNumFood()

    eval = 0
    food_count = 0
    for i in range(food_map.width):
        for j in range(food_map.height):
            if food_map[i][j]:
                food_count += 1
                food_distance = find2DDistance((i,j), (pos))
                if food_distance == 0:
                    food_distance = 0.5
                eval += 7/(food_distance+1) - 14

    # return food_map[food_map.width - 1]
    # print("RETURNS: " + str(eval + current_val) + "FOOD COUNT: " + str(food_count))

    if food_count == 0:
        return float("inf")

    return eval + current_val


# def stay_out_ghost(current_game_state):


def cleanup(currentGameState, eval):

    if currentGameState.getNumFood() > 1:
        return eval
    # This function only kicks in when there exist one food remaining in game

    # Find where the food is

    # print(currentGameState.getNumFood())
    currFood = currentGameState.getFood()
    pos = currentGameState.getPacmanPosition()
    for x in range(currFood.width):
        for y in range(currFood.height):
            if currFood[x][y]:
                return 1/ find2DDistance((x,y), pos)

    return eval



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Function does (primary) two things: 1. runs when ghost is too close
      2. tries to eat as much food as possible when ghost is not a threat.

      ( There is some loose logic that allows the pacman to eat the ghost, if
      conditions permit)
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()

    eval = 0
    eval = food_eval(currentGameState, eval)
    if ghost_too_close_technique(newGhostStates, newPos, 2, 1):
        return -float("inf")
    return eval

# Abbreviation
better = betterEvaluationFunction
