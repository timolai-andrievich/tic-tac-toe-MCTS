import random
from typing import Dict
from Game import Position, Game
from MCTS import MCTS
from NN import NN
import numpy as np
from config import Config

class Player:
    def get_action(self, game: Game) -> int:
        pass

class RandomPlayer(Player):
    def get_action(self, game: Game) -> int:
        return np.random.randint(Game.num_actions)

class MinMaxPlayer(Player):
    def __init__(self):
        self.cache: Dict[str, int] = {}
    
    def evaluate(self, game: Game) -> int:
        image = game.position.to_image()
        if image in self.cache:
            return self.cache[image]
        if game.is_terminal():
            self.cache[image] = game.get_winner()
        else:
            res = None
            for action in game.get_actions():
                scratch_game = game.copy()
                scratch_game.commit_action(action)
                if res is None or res < game.get_current_move() * self.evaluate(scratch_game):
                    res = game.get_current_move() * self.evaluate(scratch_game)
            self.cache[image] = res
        return self.cache[image]

    def get_action(self, game: Game) -> int:
        eval = self.evaluate(game)
        moves = []
        for action in game.get_actions():
            scratch_game = game.copy()
            scratch_game.commit_action(action)
            if self.evaluate(scratch_game) == eval:
                moves.append(action)
        return random.choice(moves)

class MctsPlayer(Player):
    def __init__(self, nn: NN, config: Config):
        self.nn = nn
        self.config = config

    def get_action(self, game: Game) -> int:
        tree = MCTS(self.config)
        actions, results = tree.run(game, self.nn.policy_function)
        action = np.argmax(actions)
        return action

class ModelPlayer(Player):
    def __init__(self, nn: NN):
        self.nn = nn

    def get_action(self, game: Game) -> int:
        actions, results = self.nn.policy_function(game.position)
        action = np.argmax(actions)
        return action
