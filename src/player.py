"""Contains classes encapsulating several game strategies.
"""
import random
from typing import Dict

import numpy as np

from game import Game
from mcts import MCTS
from policy import Model
from config import Config


class Player:  # pylint: disable=too-few-public-methods
    """Abstract class, represents a playing strategy.
    """

    def get_action(self, game: Game) -> int:
        """Chooses action based on the class strategy.

        Args:
            game (Game)

        Returns:
            int: The ID of the chosen action.
        """


class RandomPlayer(Player):  # pylint: disable=too-few-public-methods
    """Chooses a random legal action for every position.
    """

    def get_action(self, game: Game) -> int:
        """Chooses random legal action.

        Args:
            game (Game)

        Returns:
            int: The ID of the chosen action.
        """
        return np.random.choice(game.get_actions())


class MinMaxPlayer(Player):
    """Chooses random action that will not lead to losing advantage based on minmax algorithm.
    """

    def __init__(self):
        self.cache: Dict[str, int] = {}

    def evaluate(self, game: Game) -> int:
        """Evaluates the current position in the game and returns 1 if X wins,
        0 if the game will result in a draw, -1 if O wins. Evaluation is based on
        assumption that both players will play optimal moves.

        Args:
            game (Game): Game to evaluate.

        Returns:
            int: Evaluation result.
        """
        image = game.position.to_image()
        if image in self.cache:
            return self.cache[image]
        if game.is_finished():
            self.cache[image] = game.get_winner()
        else:
            res = None
            for action in game.get_actions():
                scratch_game = game.copy()
                scratch_game.commit_action(action)
                if res is None or res * game.get_current_move(
                ) < game.get_current_move() * self.evaluate(scratch_game):
                    res = self.evaluate(scratch_game)
            self.cache[image] = res
        return self.cache[image]

    def get_action(self, game: Game) -> int:
        """Chooses random optial action.

        Args:
            game (Game)

        Returns:
            int: The ID of the chosen action.
        """
        evaluation = self.evaluate(game)
        moves = []
        for action in game.get_actions():
            scratch_game = game.copy()
            scratch_game.commit_action(action)
            if self.evaluate(scratch_game) == evaluation:
                moves.append(action)
        return random.choice(moves)


class MctsPlayer(Player):  # pylint: disable=too-few-public-methods
    """Chooses action based on Monte-Carlo Tree Search algorithm.
    Uses neural network as a policy function.
    """

    def __init__(self, model: Model, config: Config):
        """Chooses action based on Monte-Carlo Tree Search algorithm.
        Uses neural network as a policy function.

        Args:
            model (Model): Neural network to be used as a policy function.
            config (Config): Parameters for model and MCTS.
        """
        self.model = model
        self.config = config

    def get_action(self, game: Game) -> int:
        """Chooses legal action using NN and MCTS.

        Args:
            game (Game)

        Returns:
            int: The ID of the chosen action.
        """
        tree = MCTS(self.config)
        actions, _ = tree.run(game.copy(), self.model.policy_function)
        action = np.argmax(actions)
        return int(action)


class ModelPlayer(Player):  # pylint: disable=too-few-public-methods
    """Chooses action based on pure neural network evaluation, without the MCTS algorithm.
    """

    def __init__(self, model: Model):
        """Chooses action based on pure neural network evaluation, without the MCTS algorithm.

        Args:
            model (Model): Neural network to be used as a policy function.
        """
        self.model = model

    def get_action(self, game: Game) -> int:
        """Chooses legal action using NN.

        Args:
            game (Game)

        Returns:
            int: The ID of the chosen action.
        """
        actions, _ = self.model.policy_function(game.position, training=False)
        legal_actions = game.get_actions()
        action = max(legal_actions, key=lambda x: actions[x])
        return action
