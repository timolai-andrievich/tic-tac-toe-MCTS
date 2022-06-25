from numpy import ndarray
from torch import le
from Game import Game, Position, NUM_ACTIONS
from typing import Tuple, Dict

c_impact = 0.2


class Node:
    """Represents a node in the Monte-Carlo search tree"""

    def __init__(self, parent, prior: float):
        self._parent: Node = parent
        self._prior: float = prior
        self._children: Dict[int, Node] = {}
        self._visits: int = 0
        self._avg: float = 0

    def select(self) -> Tuple[int, Node]:
        """Selects the node with the best UCB score, and returns action leading to that node and the Node itself"""
        return max(self._children.items(), key=lambda x: x[1].value())

    def expand(self, game: Game, action_probs: ndarray) -> None:
        """Expands the node"""
        legal_actions = set(game.get_actions())
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


class MCST:
    """Represents the search tree"""

    def __init__(self, game: Game, policy_function):
        self._game: Game = game
        self._policy = policy_function
        self._root = Node(None, 0)

    def run(self, game: Game) -> Dict[int, float]:
        """Returns the list of probabilities of actions"""

    def simulate(self, game: Game, policy_function) -> int:
        """Simulates the game and returns the winner"""

    def commit_action(self, action: int):
        """Makes the subtree of the root corresponding to the action the new root, and discards all the other nodes"""
