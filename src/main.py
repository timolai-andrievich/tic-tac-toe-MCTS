from typing import List, Tuple

import numpy as np
import tqdm
from numpy import ndarray

from Game import Game, Image
from MCTS import MCTS
from NN import NN
from config import Config
from player import RandomPlayer
from utils import evaluate_pure_models_against_player


def train(nn: NN, config: Config):
    nn.update_config(config)
    exploration_noise = config.starting_exploration_noise
    for i in tqdm.tqdm(range(config.iteration_count)):
        training_data: List[Tuple[Image, Tuple[ndarray, ndarray]]] = []
        for _ in tqdm.tqdm(range(config.games_in_iteration)):
            game = Game().copy()
            debug_x = game.is_terminal()
            if debug_x:
                pass
            tree = MCTS(config)
            while not game.is_terminal():
                probs, wdl = tree.run(game.copy(), nn.policy_function)
                probs = probs * (
                        1 - exploration_noise
                ) + exploration_noise * np.random.dirichlet(np.ones(Game.num_actions))
                legal_actions = game.get_actions()
                for a in range(Game.num_actions):
                    if not a in legal_actions:
                        probs[a] = 0
                probs = probs / probs.sum()
                probs = np.power(probs, 1 / config.temp)
                probs = probs / probs.sum()
                action = np.random.choice(Game.num_actions, p=probs)
                training_data.append((game.position.to_image(), (probs, wdl)))
                game.commit_action(action)
                tree.commit_action(action)
        batch = training_data
        nn.train(config, batch)
        if i > 0 and i % config.checkpoints == 0:
            nn.dump(info=f"iteration_{i}")
        if i % config.test_checkpoints == 0:
            pass
        exploration_noise *= config.exploration_decay
        if exploration_noise < config.min_exploration_noise:
            exploration_noise = config.min_exploration_noise
    nn.dump()
    return nn


def main():
    config = Config()
    config.learning_rate = 2e-3
    config.games_in_iteration = 50
    config.mcts_playout = 50
    config.iteration_count = 50
    config.starting_exploration_noise = 0.5
    config.min_exploration_noise = 0.1
    config.exploration_decay = 0.95
    nn = NN(config)
    nn = train(nn, config)
    config.starting_exploration_noise = 0.25
    config.learning_rate = 2e-4
    config.iteration_count = 50
    nn = train(nn, config)
    config.starting_exploration_noise = 0.15
    config.min_exploration_noise = 0.01
    config.learning_rate = 2e-5
    config.iteration_count = 50
    nn = train(nn, config)
    random_player = RandomPlayer()
    evaluate_pure_models_against_player(config, random_player, 500)


if __name__ == "__main__":
    main()
