from typing import List, Tuple
from typing_extensions import Self

import numpy as np
import tqdm
from numpy import ndarray

from Game import Game, Image
from MCTS import MCTS
from NN import NN
from config import Config
from player import RandomPlayer
from utils import evaluate_pure_models_against_player

class SelfplayGenerator:
    def __init__(self, nn: NN, config: Config):
        self.nn = nn
        self.config = config
        self.x = np.zeros((0, Game.board_height, Game.board_width, Game.num_layers))
        self.y_act = np.zeros((0, Game.num_actions))
        self.y_wdl = np.zeros((0, 3))

    
    def generate_games(self, count: int) -> Tuple[ndarray, ndarray, ndarray]:
        for _ in range(count):
            self.generate_game()
        return self.x, self.y_act, self.y_wdl

    
    def generate_game(self):
        game = Game()
        x: ndarray = np.zeros((self.config.max_moves, Game.board_height, Game.board_width, Game.num_layers))
        y_act: ndarray = np.zeros((self.config.max_moves, Game.num_actions))
        y_wdl: ndarray = np.zeros((self.config.max_moves, 3))
        tree = MCTS(self.config)
        current_move = 0
        while not game.is_terminal():
            probs, wdl = tree.run(game.copy(), self.nn.policy_function)
            probs = probs * (
                    1 - self.config.exploration_noise
            ) + self.config.exploration_noise * np.random.dirichlet(np.ones(Game.num_actions))
            legal_actions = game.get_actions()
            for a in range(Game.num_actions):
                if not a in legal_actions:
                    probs[a] = 0
            probs = probs / probs.sum()
            probs = np.power(probs, 1 / self.config.temp)
            probs = probs / probs.sum()
            action = np.random.choice(Game.num_actions, p=probs)
            x[current_move] = game.position.vectorize()
            y_act[current_move] = probs
            y_wdl[current_move] = wdl
            current_move += 1
            game.commit_action(action)
            tree.commit_action(action)
        self.x = np.append(self.x, x, axis=0)
        self.y_act = np.append(self.y_act, y_act, axis=0)
        self.y_wdl = np.append(self.y_wdl, y_wdl, axis=0)
    



def train(nn: NN, config: Config):
    nn.update_config(config)
    exploration_noise = config.starting_exploration_noise
    for i in tqdm.tqdm(range(config.iteration_count)):
        training_data: List[Tuple[Image, Tuple[ndarray, ndarray]]] = []
        generator = SelfplayGenerator(nn, config)
        training_data = generator.generate_games(config.games_in_iteration)
        nn.train(config, training_data)
        if i > 0 and i % config.checkpoints == 0:
            nn.dump(info=f"iteration_{i}")
        if i % config.test_checkpoints == 0:
            pass
        config.exploration_noise *= config.exploration_decay
        if config.exploration_noise < config.min_exploration_noise:
            config.exploration_noise = config.min_exploration_noise
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
