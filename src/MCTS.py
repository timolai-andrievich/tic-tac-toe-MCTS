from Game import Game, Position, NUM_ACTIONS
from typing import Tuple, Dict


class Node:
    """Represents a node in the Monte-Carlo search tree"""

    def __init__(self, parent: Node, prior: float):
        self._prior: float = prior
        self._children: Dict[int, Node] = {}
        self._visits: int = 0

    def select(self) -> Tuple[int, Node]:
        """Selects the node with the best UCB score, and returns action leading to that node and the Node itself"""

    def expand(self, game: Game) -> None:
        """Expands the node"""

    def is_leaf(self) -> bool:
        """Returns True if the node is a leaf, false otherwise"""

    def is_root(self):
        """Returns True if the node is the root of the tree, false otherwise"""

    def ucb_score(self) -> float:
        """Returns the UCB score of the node"""


class MCST:
    """Represents the search tree"""

    def __init__(self, game: Game, policy_function):
        self._game = game
        self._policy = policy_function
        self._root = Node(None, 0)

    def run(self, game: Game) -> Dict[int, float]:
        """Returns the list of probabilities of actions"""

    def simulate(self, game: Game, policy_function) -> int:
        """Simulates the game and returns the winner"""
