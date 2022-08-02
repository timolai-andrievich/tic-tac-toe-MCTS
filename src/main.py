"""Entry point of the program.
"""
from typing import Tuple

import numpy as np
import tqdm
from numpy import ndarray

from game import Game
from mcts import MCTS
from policy import Model
from config import Config
from player import RandomPlayer
from utils import evaluate_pure_models_against_player


class SelfplayGenerator:
    """Generates games through self-play.
    """

    def __init__(self, model: Model, config: Config):
        """Generates games through self-play.

        Args:
            model (Model): Model to generate games with.
            config (Config): Parameters used for generation.
        """
        self.model = model
        self.config = config
        self.state_tensor = np.zeros(
            (0, Game.board_height, Game.board_width, Game.num_layers))
        self.y_act = np.zeros((0, Game.num_actions))
        self.y_wdl = np.zeros((0, 3))

    def generate_games(self, count: int) -> Tuple[ndarray, ndarray, ndarray]:
        """Generates games through self-play, and returns the data about generated games.

        Args:
            count (int): The amount to games to play.

        Returns:
            Tuple[ndarray, ndarray, ndarray]: Returns tensors with game states,
            action probabilities, and outcome probabilities
        """
        for _ in range(count):
            self.generate_game()
        return self.state_tensor, self.y_act, self.y_wdl

    def generate_game(self):
        """Generates a game and adds it to the class attributes.
        """
        game = Game()
        states: ndarray = np.zeros((self.config.max_moves, Game.board_height,
                                    Game.board_width, Game.num_layers))
        y_act: ndarray = np.zeros((self.config.max_moves, Game.num_actions))
        y_wdl: ndarray = np.zeros((self.config.max_moves, 3))
        # Initialize Monte-Carlo Tree Search
        tree = MCTS(self.config)
        current_move = 0
        while not game.is_finished() and current_move < self.config.max_moves:
            # Get action and outcome probabilities using MCTS
            probs, wdl = tree.run(game.copy(), self.model.policy_function)
            # Add exploration noise to the actions
            probs = probs * (
                1 - self.config.exploration_noise
            ) + self.config.exploration_noise * np.random.dirichlet(
                np.ones(Game.num_actions))
            # Remove all illegal actions from probabilities
            legal_actions = game.get_actions()
            for action in range(Game.num_actions):
                if not action in legal_actions:
                    probs[action] = 0
            # Apply temperature hyperparameter to the action probabilities
            probs = probs / probs.sum()
            probs = np.power(probs, 1 / self.config.temp)
            probs = probs / probs.sum()
            action = np.random.choice(Game.num_actions, p=probs)
            states[current_move] = game.position.get_state()
            y_act[current_move] = probs
            y_wdl[current_move] = wdl
            current_move += 1
            game.commit_action(action)
            tree.commit_action(action)
        self.state_tensor = np.append(self.state_tensor,
                                      states[:current_move],
                                      axis=0)
        self.y_act = np.append(self.y_act, y_act[:current_move], axis=0)
        self.y_wdl = np.append(self.y_wdl, y_wdl[:current_move], axis=0)


def train(model: Model, config: Config, checkpoints=False) -> Model:
    """Train the given model.

    Args:
        model (Model): Model to train.
        config (Config): Parameters for training and MCTS.
        checkpoints (bool, optional): Save intermediate results to checkpoints. Defaults to False.

    Returns:
       Model: Trained model.
    """
    model.update_config(config)
    config.exploration_noise = config.starting_exploration_noise
    for i in tqdm.tqdm(range(config.iteration_count)):
        generator = SelfplayGenerator(model, config)
        training_data = generator.generate_games(config.games_in_iteration)
        model.train(config, training_data)
        if checkpoints and i > 0 and i % config.checkpoints == 0:
            model.save(info=f"iteration_{i}")
        config.exploration_noise *= config.exploration_decay
        if config.exploration_noise < config.min_exploration_noise:
            config.exploration_noise = config.min_exploration_noise
    return model


def main():
    """The entry function of the application.
    """
    config = Config()
    config.learning_rate = 2e-3
    config.games_in_iteration = 50
    config.mcts_playout = 20
    config.iteration_count = 30
    config.starting_exploration_noise = 0.5
    config.min_exploration_noise = 0.1
    config.exploration_decay = 0.95
    model = Model(config)
    model = train(model, config)
    config.starting_exploration_noise = 0.25
    config.learning_rate = 2e-4
    config.iteration_count = 30
    model = train(model, config)
    config.starting_exploration_noise = 0.15
    config.min_exploration_noise = 0.01
    config.learning_rate = 2e-5
    config.iteration_count = 10
    model = train(model, config)
    model.save()
    random_player = RandomPlayer()
    evaluate_pure_models_against_player(config, random_player, 1000)


if __name__ == "__main__":
    main()
