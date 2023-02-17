"""Entry point of the program.
"""
from typing import Tuple, Optional
import os.path

import numpy as np
import tqdm
from numpy import ndarray

from game import Game
from mcts import MCTS
from policy import Model
from config import Config
from player import RandomPlayer
from utils import evaluate_pure_models_against_player

required_folders = ['../models']


class SelfplayGenerator:
    """Generates games through self-play.
    """

    def __init__(self, model: Model, config: Config):
        """Generates games through self-play.

        Args:
            model (Model): Model to generate games with.
            config (Config): Parameters used for generation.
        """
        self.game_idx = 0
        self.model = model
        self.config = config
        self.states = np.zeros(
            (0, Game.board_height, Game.board_width, Game.num_layers))
        self.actions = np.zeros((0, Game.num_actions))
        self.wdl = np.zeros((0, 3))

    def generate_games(self, count: int):
        """Generates `count` games through self-play, and adds them to the internal buffer.

        Args:
            count (int): The amount to games to play.
        """
        for self.game_idx in range(count):
            self.generate_game()
        if len(self.states) > self.config.buffer_size:
            mask = np.ones(len(self.states), np.bool8)
            mask[:-self.config.buffer_size] = 0
            self.states = self.states[mask]
            self.actions = self.actions[mask]
            self.wdl = self.wdl[mask]

    def generate_game(self):
        """Generates a game and adds it to the class attributes.
        """
        game = Game()
        current_game_states: ndarray = np.zeros((self.config.max_moves, Game.board_height,
                                    Game.board_width, Game.num_layers))
        current_game_actions: ndarray = np.zeros((self.config.max_moves, Game.num_actions))
        current_game_wdl: ndarray = np.zeros((self.config.max_moves, 3))
        # Initialize Monte-Carlo Tree Search
        tree = MCTS(self.config)
        current_move = 0
        while not game.is_finished() and current_move < self.config.max_moves:
            # Get action and outcome probabilities using MCTS
            probs, wdl = tree.run(game.copy(), self.model.policy_function)
            # Apply temperature hyperparameter to the action probabilities
            probs = np.power(probs, 1 / self.config.temp)
            probs = probs / probs.sum()
            # Save probabilities to model training
            pure_probs = probs.copy()
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
            probs = probs / probs.sum()
            action = np.random.choice(Game.num_actions, p=probs)
            current_game_states[current_move] = game.position.get_state()
            current_game_actions[current_move] = pure_probs
            current_game_wdl[current_move] = wdl
            current_move += 1
            game.commit_action(action)
            tree.commit_action(action)
        self.states = np.append(self.states,
                                      current_game_states[:current_move],
                                      axis=0)
        self.actions = np.append(self.actions, current_game_actions[:current_move], axis=0)
        self.wdl = np.append(self.wdl, current_game_wdl[:current_move], axis=0)

    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[ndarray, ndarray, ndarray]:
        """Sample a batch from the internal buffer and return the tuple (states, actions, WDL).

        Args:
            batch_size (Optional[int]): The size of the batch to sample. If not provided,
            defaults to the batch size specified in config

        Returns:
            Tuple[ndarray, ndarray, ndarray]: The tuple (states, actions, WDL).
        """
        assert len(self.states) > 0
        if batch_size is None:
            batch_size = self.config.batch_size
        indexes = np.random.choice(len(self.states), batch_size)
        return self.states[indexes], self.actions[indexes], self.wdl[indexes]


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
    generator = SelfplayGenerator(model, config)
    for i in tqdm.tqdm(range(config.iteration_count)):
        generator.generate_games(config.games_in_iteration)
        for i in range(config.epochs):
            training_data = generator.get_batch()
            model.train(config, training_data)
        if checkpoints and i % config.checkpoints_interval == 0:
            model.save(info=f"iteration_{i + 1}")
        config.exploration_noise *= config.exploration_decay
        if config.exploration_noise < config.min_exploration_noise:
            config.exploration_noise = config.min_exploration_noise
    return model


def main():
    """The entry function of the application.
    """
    for path in required_folders:
        if not os.path.exists(path):
            os.mkdir(path)
    config = Config()
    model = Model(config)
    model = train(model, config)
    model.save()
    random_player = RandomPlayer()
    evaluate_pure_models_against_player(config, random_player, 100)


if __name__ == "__main__":
    main()
