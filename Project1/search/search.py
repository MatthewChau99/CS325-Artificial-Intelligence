###
#   THIS CODE WAS MY OWN WORK , IT WAS WRITTEN WITHOUT CONSULTING ANY
#    SOURCES OUTSIDE OF THOSE APPROVED BY THE INSTRUCTOR. Matthew Chau
###

# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    open = util.Stack()
    closed = set()
    current, actions = problem.getStartState(), []
    open.push((current, actions))

    while not open.isEmpty():
        current, actions = open.pop()
        if problem.isGoalState(current):
            return actions
        closed.add(current)
        for successor in problem.getSuccessors(current):  # successor is a tuple (successor, action, stepCost)
            if successor[0] not in closed:
                open.push((successor[0], actions + [successor[1]]))

    raise Exception("Cannot find path")


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    open = util.Queue()
    closed = set()
    current, actions = problem.getStartState(), []
    open.push((current, actions))

    while not open.isEmpty():
        current, actions = open.pop()
        if problem.isGoalState(current):
            return actions
        closed.add(current)
        for successor in problem.getSuccessors(current):  # successor is a tuple (successor, action, stepCost)
            if successor[0] not in closed:
                open.push((successor[0], actions + [successor[1]]))
                closed.add(successor[0])

    raise Exception("Cannot find path")


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    open = util.PriorityQueue()
    closed = set()
    current, actions, cost = problem.getStartState(), [], 0
    open.push((current, actions, cost), cost)

    while not open.isEmpty():
        # pop until 'current' is an unvisited state
        while current in closed:
            current, actions, cost = open.pop()
        if problem.isGoalState(current):
            return actions
        closed.add(current)
        for successor in problem.getSuccessors(current):
            if successor not in closed:
                open.push((successor[0], actions + [successor[1]], cost + successor[2]), cost + successor[2])

    raise Exception("Cannot find path")


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    open = util.PriorityQueue()
    closed = set()
    current, actions, cost = problem.getStartState(), [], 0
    open.push((current, actions, cost), cost)

    while not open.isEmpty():
        while current in closed:
            current, actions, cost = open.pop()
        if problem.isGoalState(current):
            return actions
        closed.add(current)
        for successor in problem.getSuccessors(current):
            if successor not in closed:
                # taking f = cost + heuristic as the key for the Priority Queue
                open.push((successor[0], actions + [successor[1]], cost + successor[2]), cost + successor[2] + heuristic(successor[0], problem))

    raise Exception("Cannot find path")


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
