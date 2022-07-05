import glob
import itertools
import numpy as np
from Game import NUM_ACTIONS, Game, Image
from NN import NN
from MCTS import MCST
from typing import Dict, List, Tuple
from numpy import ndarray
import tqdm
import random

iteration_count = 1
games_in_iteration = 1
mcts_playout = 2
batch_size = -1
checkpoints = 10
test_games = 1
starting_exploration_noise = 1
exploration_decay = 0.95
min_exploration_noise = 0.15


def validate_parameters():
    assert 1 > exploration_decay >= 0
    assert 1 > starting_exploration_noise >= 0
    assert 1 > min_exploration_noise >= 0
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


def models_play(nn1: NN, nn2: NN, first_starts: bool):
    """Returns 1 if the first nn wins, 0 if the game is tied, -1 if thesecond nn wins"""
    game = Game().copy()
    tree1 = MCST()
    tree2 = MCST()
    if first_starts:
        trees = [tree1, tree2]
        policies = [nn1.policy_function, nn2.policy_function]
    else:
        trees = [tree2, tree1]
        policies = [nn2.policy_function, nn1.policy_function]
    i: int = 0
    while not game.is_terminal():
        probs, _ = trees[i & 1].run(game.copy(), policies[i & 1], mcts_playout)
        legal_actions = game.get_actions()
        for a in range(NUM_ACTIONS):
            if not a in legal_actions:
                probs[a] = 0
        probs = probs / probs.sum()
        action = np.argmax(probs)
        trees[0].commit_action(action)
        trees[1].commit_action(action)
        game.commit_action(action)
        i += 1
    winner = game.get_winner()
    if winner == 0:
        res = 0
    else:
        res = (1 if first_starts else -1) * winner
    return res


def models_match(nn1: NN, nn2: NN, games: int = test_games):
    res = [0, 0, 0]
    for i in range(games):
        winner = models_play(nn1, nn2, i % 2 == 0)
        res[winner] += 1
    return res


def models_tournament_round(silent=False):
    files = zip(glob.glob("../models/*policy*"), glob.glob("../models/*value*"))
    models_list: List[str] = []
    models_results: Dict[str, List[int, int, int]] = {}
    models: Dict[str, NN] = {}
    for policy, value in files:
        nn = NN(value_file=value, policy_file=policy)
        models_list.append(policy)
        models[policy] = nn
        models_results[policy] = [0, 0, 0]
    models_list.append("random")
    models["random"] = NN()
    models_results["random"] = [0, 0, 0]
    matchups = itertools.combinations(models_list, 2)
    for file1, file2 in tqdm.tqdm(list(matchups)):
        nn1 = models[file1]
        nn2 = models[file2]
        res = models_match(nn1, nn2)
        t, w, l = res
        print(f"{file1} - {file2}: +{w}-{l}={t}")
        models_results[file1][0] += res[0]
        models_results[file1][1] += res[1]
        models_results[file1][-1] += res[-1]

        models_results[file2][0] += res[0]
        models_results[file2][1] += res[-1]
        models_results[file2][-1] += res[1]
    sorted_models = list(
        sorted(models_results.items(), key=lambda x: x[1][-1] - x[1][1])
    )
    if not silent:
        for model, (t, w, l) in sorted_models:
            print(f"{model}: +{w}-{l}={t}")
    return sorted_models


def profile():
    import pstats
    import cProfile

    with cProfile.Profile() as p:
        train()
    stats = pstats.Stats(p)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()


def main():
    train()


if __name__ == "__main__":
    main()
