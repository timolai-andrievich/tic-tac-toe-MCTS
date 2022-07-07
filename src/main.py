import glob
import itertools
import numpy as np
from torch import rand
from Game import NUM_ACTIONS, Game, Image
from NN import NN
from MCTS import MCST
from typing import Dict, List, Tuple
from numpy import ndarray
import tqdm
import random
from utils import models_tournament_round

iteration_count = 1
games_in_iteration = 1
mcts_playout = 100
batch_size = -1
checkpoints = 10
test_games = 50
starting_exploration_noise = 1
exploration_decay = 0.95
min_exploration_noise = 0.15


def validate_parameters():
    assert 1 > exploration_decay >= 0
    assert 1 >= starting_exploration_noise >= 0
    assert 1 >= min_exploration_noise >= 0
    assert mcts_playout > 1  # Otherwise, the root of the tree will have no children


validate_parameters()


def make_batch(
    training_data: List[Tuple[Image, Tuple[ndarray, float]]],
    batch_size: int = batch_size,
) -> List[Tuple[Image, Tuple[ndarray, float]]]:
    if batch_size == -1:
        return training_data
    batch: List[Tuple[Image, Tuple[ndarray, float]]] = []
    for _ in range(batch_size):
        batch.append(random.choice(training_data))
    return batch


def train(policy_file=None, value_file=None):
    nn = NN(policy_file=policy_file, value_file=value_file)
    exploration_noise = starting_exploration_noise
    for i in tqdm.tqdm(range(iteration_count)):
        training_data: List[Tuple[Image, Tuple[ndarray, float]]] = []
        for _ in range(games_in_iteration):
            game = Game().copy()
            debug_x = game.is_terminal()
            if debug_x:
                pass
            tree = MCST()
            while not game.is_terminal():
                probs, eval = tree.run(game.copy(), nn.policy_function, mcts_playout)
                probs = probs * (
                    1 - exploration_noise
                ) + exploration_noise * np.random.dirichlet(np.ones(NUM_ACTIONS))
                legal_actions = game.get_actions()
                for a in range(NUM_ACTIONS):
                    if not a in legal_actions:
                        probs[a] = 0
                probs = probs / probs.sum()
                action = np.random.choice(NUM_ACTIONS, p=probs)
                training_data.append((game._position.to_image(), (probs, eval)))
                game.commit_action(action)
                tree.commit_action(action)
        batch = make_batch(training_data)
        nn.train(batch)
        if i > 0 and i % checkpoints == 0:
            nn.dump(info=f"iteration_{i}")
        exploration_noise *= exploration_decay
        if exploration_noise < min_exploration_noise:
            exploration_noise = min_exploration_noise
    nn.dump()
    return nn


def main():
    models_tournament_round(action_choice='random_sharp')

if __name__ == "__main__":
    main()
