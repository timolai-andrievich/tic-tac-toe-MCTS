import glob
import itertools
import time
from unittest import result
import numpy as np
from torch import rand
from Game import BOARD_HEIGHT, Game, Image
from NN import NN
from MCTS import MCST
from typing import Dict, List, Tuple
from numpy import ndarray
import tqdm
import random
from colorama import Fore


action_choices = {"best", "random", "probabilities", "random_sharp"}
mcts_playout = 100
test_games = 20


class TrueRandom(NN):
    def __init__(*args, **kwargs):
        pass

    def train(*args, **kwargs):
        pass

    def policy_function(*args, **kwargs):
        return (
            np.random.dirichlet(np.ones(Game.num_actions)),
            np.random.dirichlet(np.ones(3)),
        )

    def dump(*args, **kwargs):
        pass


class EqualProbs(NN):
    def __init__(*args, **kwargs):
        pass

    def train(*args, **kwargs):
        pass

    def policy_function(*args, **kwargs):
        return (
            np.ones(Game.num_actions) / Game.num_actions,
            np.ones(3) / 3,
        )

    def dump(*args, **kwargs):
        pass



def models_play(nn1: NN, nn2: NN, first_starts: bool, action_choice="best"):
    """Returns 1 if the first nn wins, 0 if the game is tied, -1 if thesecond nn wins"""
    assert action_choice in action_choices
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
        for a in range(Game.num_actions):
            if not a in legal_actions:
                probs[a] = 0
        probs = probs / probs.sum()
        if action_choice == "best":
            action = np.argmax(probs)
        elif action_choice == "probabilities":
            action = np.random.choice(Game.num_actions, p=probs)
        elif action_choice == "random":
            action = random.choice(legal_actions)
        elif action_choice == "random_sharp":
            p = probs - np.min(probs)
            p /= p.sum()
            action = np.random.choice(Game.num_actions, p=p)
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


def models_match(nn1: NN, nn2: NN, games: int = test_games, action_choice="best"):
    assert action_choice in action_choices
    res = [0, 0, 0]
    for i in tqdm.tqdm(range(games)):
        winner = models_play(nn1, nn2, i % 2 == 0, action_choice=action_choice)
        res[winner] += 1
    return res


def models_tournament_round(silent=False, action_choice="best"):
    assert action_choice in action_choices
    # TODO add tournament table visualization
    files = glob.glob("../models/*")
    models_list: List[str] = []
    models_results: Dict[str, List[int, int, int]] = {}
    models: Dict[str, NN] = {}
    for file_path in files:
        nn = NN(file_path=file_path)
        models_list.append(file_path)
        models[file_path] = nn
        models_results[file_path] = [0, 0, 0]
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


def play_and_visualize(nn1: NN, nn2: NN, action_choice='best'):
    assert action_choice in action_choices
    game = Game().copy()
    tree1 = MCST()
    tree2 = MCST()
    trees = [tree1, tree2]
    policies = [nn1.policy_function, nn2.policy_function]
    i: int = 0
    t1 = time.time()
    print(game.position.visualize(), '\n')
    try:
        while not game.is_terminal():
            probs, results = trees[i & 1].run(game.copy(), policies[i & 1], mcts_playout)
            legal_actions = game.get_actions()
            for a in range(Game.num_actions):
                if not a in legal_actions:
                    probs[a] = 0
            probs = probs / probs.sum()
            if action_choice == "best":
                action = np.argmax(probs)
            elif action_choice == "probabilities":
                action = np.random.choice(Game.num_actions, p=probs)
            elif action_choice == "random":
                action = random.choice(legal_actions)
            elif action_choice == "random_sharp":
                p = probs - np.min(probs)
                p /= p.sum()
                action = np.random.choice(Game.num_actions, p=p)
            trees[0].commit_action(action)
            trees[1].commit_action(action)
            game.commit_action(action)
            print(f"Making move {action} with probability {probs[action] * 100:.2f}%"
                f" outcome probabilities: X - {results[1]:.2f}, tie - {results[0]:.2f}, O - {results[2]:.2f}")
            pos = list(game.position.visualize())
            c = pos[action // Game.board_height * (Game.board_height + 1) + action % Game.board_height]
            c = Fore.RED + c + Fore.RESET
            pos[action // Game.board_height * (Game.board_height + 1) + action % Game.board_height] = c
            print(''.join(pos))
            i += 1
        t2 = time.time()
        winner = game.get_winner()
        if winner == 0:
            print('The game is a tie')
        elif winner == 1:
            print('X won')
        else:
            print('O won')
        print(f'{i} moves played in {t2 - t1:.2f}s, {(t2 - t1) / i:.2f}s/move')
    except KeyboardInterrupt:
        pass