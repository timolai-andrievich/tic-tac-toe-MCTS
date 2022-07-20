from typing import Tuple, Dict, List
import math

from numpy import ndarray
import numpy as np

from config import Config
from game import Game

probs_to_eval = np.array([0, 1, -1])


class Node:
    """Represents the node of the game tree.
    """

    def __init__(self, parent, prior: float, current_move: int, config: Config):
        """Represents the node of the game tree

        Args:
            parent (Node): Parent of the node.
            prior (float)
            current_move (int): The player that is to move
            from the position represented by the node.
            config (Config): _description_
        """
        self.config = config
        self.current_move: int = current_move
        self.parent: Node = parent
        self.prior: float = prior
        self.children: Dict[int, Node] = {}
        self.visits: int = 0
        self.results = np.array([0.0, 0.0, 0.0])

    def avg(self) -> float:
        """Returns the numerical evaluation of the position.

        Returns:
            float: Evaluation.
        """
        if abs(self.results.sum()) < 1e-6:
            return 0
        return (self.results / self.results.sum()).dot(probs_to_eval)

    def select(self) -> Tuple[int, any]:
        """Selects the child of the node and returns
        the corresponding action and the child Node object.

        Returns:
            Tuple[int, Node]: Action and the selected node.
        """
        res = None
        max_value = 0
        for action, node in self.children.items():
            if res is None or max_value < node.ucb_score():
                res = action
                max_value = node.ucb_score()
        return res, self.children[res]

    def expand(self, game: Game, action_probs: ndarray) -> None:
        """Expands the node according to the action probabilities.

        Args:
            game (Game): Game object with the current position.
            action_probs (ndarray): Probability distribution of actions.
        """
        legal_actions = game.get_actions()
        action_probs = action_probs.reshape(Game.num_actions)
        for i in legal_actions:
            self.children[i] = Node(self, action_probs[i],
                                    self.current_move * -1, self.config)

    def is_leaf(self) -> bool:
        """Retruns True if a node is a leaf, False otherwise.

        Returns:
            bool: The result of the function.
        """
        return self.children == {}

    def is_root(self) -> bool:
        """Returns True if the node is the root of the tree, False otherwise.

        Returns:
            bool: The result of the function.
        """
        return self.parent is None

    def ucb_score(self) -> float:
        """UCB(Upper Confidence Bound) score of the node.

        Returns:
            float: UCB value of the node.
        """
        return self.avg(
        ) * self.current_move * -1 + self.config.c_impact * self.prior * math.sqrt(
            self.parent.visits) / (1 + self.visits)

    def update(self, new_score: ndarray):
        """Updates the accumulative score of the node.

        Args:
            new_score (ndarray): Tie/First player wins/Second player wins probability distribution.
        """
        self.visits += 1
        self.results += new_score

    def update_recursive(self, new_score):
        """Recursively updates the accumulative score of the node, untill the root is reached.

        Args:
            new_score (ndarray): Tie/First player wins/Second player wins probability distribution.
        """
        self.update(new_score)
        if not self.is_root():
            self.parent.update_recursive(new_score)


class MCTS:
    """Class that implements Monte-Carlo Tree Search
    """

    def __init__(self, config: Config):
        """Class that implements Monte-Carlo Tree Search

        Args:
            config (Config): Parameters for MCTS.
        """
        self.config = config
        self.root = Node(None, 0, 1, self.config)

    def run(self, game: Game, policy_function) -> Tuple[ndarray, ndarray]:
        """Runs the MCTS and returns probabilities of actions and outcomes.

        Args:
            game (Game): Game object. The current position will be the root of the tree.
            policy_function: Function that takes in a position and returns action probabilities and
            outcome probabilities.

        Returns:
            Tuple[ndarray, ndarray]: Probabilities of actions and outcomes.
        """
        for _ in range(self.config.mcts_playout):
            self.simulate(game.copy(), policy_function)
        visits = np.array([
            self.root.children[i].visits if i in self.root.children else 0
            for i in range(Game.num_actions)
        ])
        probs = visits / max(1, visits.sum())
        return probs, (self.root.results / self.root.results.sum())

    def simulate(self, game: Game, policy_function):
        """Runs one step of the MCTS. Selects a leaf node, expands it,
         and then backpropagades the results.

        Args:
            game (Game): Game object. The current position will be the root of the tree.
            policy_function: Function that takes in a position and returns action probabilities and
            outcome probabilities.
        """
        actions: List[int] = []
        node: Node = self.root
        # Select a leaf node
        while not node.is_leaf():
            action, node = node.select()
            actions.append(action)
            game.commit_action(action)
        # Expand the node if possible
        if not game.is_finished():
            probs, new_value = policy_function(game.position)
            node.expand(game, probs)
        else:
            winner = game.get_winner()
            new_value = np.zeros(3)
            new_value[winner] = 1
        # Backpropagade the result
        node.update_recursive(new_value)

    def commit_action(self, action: int):
        """Makes a node corresponding to the given action the root of the tree
        and discard all the other nodes.

        Args:
            action (int): Action ID.
        """
        if action not in self.root.children:
            self.root = Node(None, 0, self.root.current_move * -1, self.config)
        else:
            self.root = self.root.children[action]
        self.root.parent = None
