from numpy import ndarray
from Game import Game
from typing import Tuple, Dict, List
import numpy as np
import math

probs_to_eval = np.array([0, 1, -1])

c_impact = 5


class Node:
    """Represents a node in the Monte-Carlo search tree"""

    def __init__(self, parent, prior: float, current_move: int):
        self.current_move: int = current_move
        self._parent: Node = parent
        self._prior: float = prior
        self._children: Dict[int, Node] = {}
        self._visits: int = 0
        self.results = np.array([0.0, 0.0, 0.0])

    def avg(self):
        if abs(self.results.sum()) < 1e-6:
            return 0
        return (self.results / self.results.sum()).dot(probs_to_eval)

    def select(self) -> Tuple[int, any]:
        """Selects the node with the best UCB score, and returns action leading to that node and the Node itself"""
        return max(self._children.items(), key=lambda x: x[1].value())

    def expand(self, game: Game, action_probs: ndarray) -> None:
        legal_actions = game.get_actions()
        action_probs = action_probs.reshape(Game.num_actions)
        for i in legal_actions:
            self._children[i] = Node(self, action_probs[i], self.current_move * -1)

    def is_leaf(self) -> bool:
        """Returns True if the node is a leaf, false otherwise"""
        return self._children == {}

    def is_root(self):
        """Returns True if the node is the root of the tree, false otherwise"""
        return self._parent == None

    def value(self) -> float:
        """Returns the UCB score of the node"""
        return self.avg() * self.current_move * -1 + c_impact * self._prior * math.sqrt(
            self._parent._visits
        ) / (1 + self._visits)

    def update(self, new_score):
        """Update the score of the node"""
        self._visits += 1
        self.results += new_score

    def update_recursive(self, new_score):
        """Update the value of the node, and the value of its parents"""
        self.update(new_score)
        if not self.is_root():
            self._parent.update_recursive(new_score)


class MCST:
    """Represents the search tree"""

    def __init__(self):
        self._root = Node(None, 0, 1)

    def run(
        self, game: Game, policy_function, num_simulations
    ) -> Tuple[ndarray, ndarray]:
        """Returns the list of probabilities of actions"""
        for _ in range(num_simulations):
            self.simulate(game.copy(), policy_function)
        visits = np.array(
            [
                self._root._children[i]._visits if i in self._root._children else 0
                for i in range(Game.num_actions)
            ]
        )
        probs = visits / max(1, np.sum(visits))
        return probs, self._root.results / self._root.results.sum()

    def simulate(self, game: Game, policy_function):
        """Simulates the game and updates the nodes of the tree"""
        actions: List[int] = []
        node: Node = self._root
        while not node.is_leaf():
            action, node = node.select()
            actions.append(action)
            game.commit_action(action)
        if not game.is_terminal():
            probs, new_value = policy_function(game.position)
            node.expand(game, probs)
        else:
            winner = game.get_winner()
            new_value = np.zeros(3)
            new_value[winner] = 1
        node.update_recursive(new_value)

    def commit_action(self, action: int):
        """Makes the subtree of the root corresponding to the action the new root, and discards all the other nodes"""
        if action not in self._root._children:
            self._root = Node(None, 0, self._root.current_move * -1)
        else:
            self._root = self._root._children[action]
        self._root._parent = None
