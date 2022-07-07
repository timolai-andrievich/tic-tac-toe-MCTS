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


action_choices = {'best', 'random', 'probabilities', 'random_sharp'}
mcts_playout = 100
test_games = 300


class TrueRandom(NN):
    def __init__(*args, **kwargs):
        pass

    def train(*args, **kwargs):
        pass

    def policy_function(*args, **kwargs):
        return np.random.dirichlet(np.ones(NUM_ACTIONS)), np.random.dirichlet(np.ones(3))
    
    def dump(*args, **kwargs):
        pass

def models_play(nn1: NN, nn2: NN, first_starts: bool, action_choice='best'):
    """Returns 1 if the first nn wins, 0 if the game is tied, -1 if thesecond nn wins"""
    assert(action_choice in action_choices)
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
        if action_choice == 'best':
            action = np.argmax(probs)
        elif action_choice == 'probabilities':
            action = np.random.choice(NUM_ACTIONS, p=probs)
        elif action_choice == 'random':
            action = random.choice(legal_actions)
        elif action_choice == 'random_sharp':
            p = probs - np.min(probs)
            p /= p.sum()
            action = np.random.choice(NUM_ACTIONS, p=p)
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


def models_match(nn1: NN, nn2: NN, games: int = test_games, action_choice='best'):
    assert(action_choice in action_choices)
    res = [0, 0, 0]
    for i in tqdm.tqdm(range(games)):
        winner = models_play(nn1, nn2, i % 2 == 0, action_choice=action_choice)
        res[winner] += 1
    return res


def models_tournament_round(silent=False, action_choice='best'):
    assert(action_choice in action_choices)
    # TODO add tournament table visualization
    files = zip(glob.glob("../models/*policy*"), glob.glob("../models/*value*"))
    models_list: List[str] = []
    models_results: Dict[str, List[int, int, int]] = {}
    models: Dict[str, NN] = {}
    for policy, value in files:
        nn = NN(value_file=value, policy_file=policy)
        models_list.append(policy)
        models[policy] = nn
        models_results[policy] = [0, 0, 0]
    models_list.append("random_nn")
    models["random_nn"] = NN()
    models_results["random_nn"] = [0, 0, 0]
    models_list.append("true_random")
    models["true_random"] = TrueRandom()
    models_results["true_random"] = [0, 0, 0]
    matchups = itertools.combinations(models_list, 2)
    for file1, file2 in tqdm.tqdm(list(matchups)):
        nn1 = models[file1]
        nn2 = models[file2]
        res = models_match(nn1, nn2, action_choice=action_choice)
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