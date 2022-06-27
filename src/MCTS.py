from hashlib import new
from numpy import ndarray
from torch import le
from Game import Game, Position, NUM_ACTIONS
from typing import Tuple, Dict, List
import numpy as np

c_impact = 5


class Node:
    """Represents a node in the Monte-Carlo search tree"""

    def __init__(self, parent, prior: float):
        self._parent: Node = parent
        self._prior: float = prior
        self._children: Dict[int, Node] = {}
        self._visits: int = 0
        self._avg: float = 0

    def select(self) -> Tuple[int, any]:
        """Selects the node with the best UCB score, and returns action leading to that node and the Node itself"""
        res = max(self._children.items(), key=lambda x: x[1].value())
        return res

    def expand(self, game: Game, action_probs: ndarray) -> None:
        """Expands the node"""
        legal_actions = set(game.get_actions())
        action_probs = action_probs.reshape(9)
        for i in legal_actions:
            self._children[i] = Node(self, action_probs[i])

    def is_leaf(self) -> bool:
        """Returns True if the node is a leaf, false otherwise"""
        return self._children == {}

    def is_root(self):
        """Returns True if the node is the root of the tree, false otherwise"""
        return self._parent == None

    def value(self) -> float:
        """Returns the UCB score of the node"""
        u = c_impact * self._prior * (self._parent._visits) ** 0.5 / (1 + self._visits)
        return self._avg + u

    def update(self, new_score):
        """Update the score of the node"""
        self._visits += 1
        self._avg += (new_score - self._avg) / self._visits

    def update_recursive(self, new_score):
        """Update the value of the node, and the value of its parents"""
        self.update(new_score)
        if not self.is_root():
            self._parent.update_recursive(-new_score)


class MCST:
    """Represents the search tree"""

    def __init__(self, game: Game, policy_function, num_simulations):
        self.num_simulations = num_simulations
        self._game: Game = game
        self._policy = policy_function
        self._root = Node(None, 0)

    def run(self, game: Game, policy_function) -> Tuple[ndarray, float]:
        """Returns the list of probabilities of actions"""
        for _ in range(self.num_simulations):
            self.simulate(game.copy(), policy_function)
        visits = np.array([self._root._children[i]._visits if i in self._root._children else 0 for i in range(9)])
        probs = visits / np.sum(visits)
        return probs, self._root._avg

    def simulate(self, game: Game, policy_function):
        """Simulates the game and updates the nodes of the tree"""
        actions: List[int] = []
        node: Node = self._root
        while not node.is_leaf():
            debug_old_node = node
            action, node = node.select()
            actions.append(action)
            game.commit_action(action)
        if not game.is_terminal():
            probs, new_value = policy_function(game._position)
            node.expand(game, probs)
        else:
            winner = game.get_winner()
            if winner == 0:
                new_value = 0
            elif winner == game.get_current_move():
                new_value = -1
            elif winner != game.get_current_move():
                new_value = 1
        node.update_recursive(new_value)

    def commit_action(self, action: int):
        """Makes the subtree of the root corresponding to the action the new root, and discards all the other nodes"""
        self._root = self._root._children[action]
        self._root._parent = None
