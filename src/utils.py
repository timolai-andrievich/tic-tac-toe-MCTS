import glob
import itertools
import random
import time
from typing import Dict, List

import numpy as np
import tqdm
from colorama import Fore
from numpy import ndarray
from scipy.stats import beta

from Game import Game
from MCTS import MCTS
from NN import NN
from minmax import MinMax
from src.config import Config

action_choices = {"best", "random", "probabilities", "random_sharp"}


def random_sharp(probs: ndarray, legal_actions):
    p = probs - np.min(probs)
    p /= p.sum()
    p += np.random.dirichlet(np.ones(p.shape)) / 100
    for i in range(Game.num_actions):
        if i not in legal_actions:
            p[i] = 0
    p /= p.sum()
    return np.random.choice(Game.num_actions, p=p)


class TrueRandom(NN):
    def __init__(self, *_, **__):
        super().__init__(Config())

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
    def __init__(self, *_, **__):
        super().__init__(Config())

    def train(*args, **kwargs):
        pass

    def policy_function(*args, **kwargs):
        return (
            np.ones(Game.num_actions) / Game.num_actions,
            np.ones(3) / 3,
        )

    def dump(*args, **kwargs):
        pass


def models_play(nn1: NN, nn2: NN, config: Config, first_starts: bool, action_choice="best"):
    """Returns 1 if the first nn wins, 0 if the game is tied, -1 if the second nn wins"""
    assert action_choice in action_choices
    game = Game().copy()
    tree1 = MCTS(config)
    tree2 = MCTS(config)
    if first_starts:
        trees = [tree1, tree2]
        policies = [nn1.policy_function, nn2.policy_function]
    else:
        trees = [tree2, tree1]
        policies = [nn2.policy_function, nn1.policy_function]
    i: int = 0
    while not game.is_terminal():
        probs, _ = trees[i & 1].run(game.copy(), policies[i & 1])
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
            action = random_sharp(probs, legal_actions)
        else:
            action = legal_actions[0]
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


def models_match(nn1: NN, nn2: NN, config: Config, action_choice="best"):
    assert action_choice in action_choices
    res = [0, 0, 0]
    for i in tqdm.tqdm(range(config.test_games)):
        winner = models_play(nn1, nn2, config, first_starts = i % 2 == 0, action_choice=action_choice)
        res[winner] += 1
    return res


def models_tournament_round(config, silent=False, action_choice="best"):
    assert action_choice in action_choices
    # TODO add tournament table visualization
    files = glob.glob("../models/*")
    models_list: List[str] = []
    models_results: Dict[str, List[int, int, int]] = {}
    models: Dict[str, NN] = {}
    for file_path in files:
        nn = NN(config, file_path=file_path)
        models_list.append(file_path)
        models[file_path] = nn
        models_results[file_path] = [0, 0, 0]
    models_list.append("random_nn")
    models["random_nn"] = NN(config)
    models_results["random_nn"] = [0, 0, 0]
    models_list.append("true_random")
    models["true_random"] = TrueRandom()
    models_results["true_random"] = [0, 0, 0]
    matchups = itertools.combinations(models_list, 2)
    for file1, file2 in tqdm.tqdm(list(matchups)):
        nn1 = models[file1]
        nn2 = models[file2]
        res = models_match(nn1, nn2, config, action_choice=action_choice)
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


def play_and_visualize(nn1: NN, nn2: NN, config: Config, action_choice='best'):
    assert action_choice in action_choices
    game = Game().copy()
    tree1 = MCTS(config)
    tree2 = MCTS(config)
    trees = [tree1, tree2]
    policies = (nn1.policy_function, nn2.policy_function)
    i: int = 0
    t1 = time.time()
    print(game.position.visualize(), '\n')
    try:
        while not game.is_terminal():
            probs, results = trees[i & 1].run(game.copy(), policies[i & 1])
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
                action = random_sharp(probs, legal_actions)
            else:
                raise ValueError(f'{action_choice} is not a valid strategy.')
            print(f"Making move {action} with probability {probs[action] * 100:.2f}%"
                  f" outcome probabilities: X - {results[1]:.2f}, tie - {results[0]:.2f}, O - {results[2]:.2f}")
            print(f'{(probs.reshape((Game.board_height, Game.board_width)) * 100).round(2)}')
            raw_probs, _ = policies[i & 1](game.position)
            print(f'{(raw_probs.reshape((Game.board_height, Game.board_width)) * 100).round(2)}')
            trees[0].commit_action(action)
            trees[1].commit_action(action)
            game.commit_action(action)
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


def play_game_against_random(nn: NN, config: Config, first_starts: bool, action_choice: str = 'best'):
    assert action_choice in action_choices
    game = Game().copy()
    tree = MCTS(config)
    policy = nn.policy_function
    i: int = 0
    while not game.is_terminal():
        if first_starts and i % 2 == 0 or not first_starts and i % 2 == 1:
            probs, _ = tree.run(game.copy(), policy)
        else:
            probs = np.random.dirichlet(np.ones(Game.num_actions))
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
            action = random_sharp(probs, legal_actions)
        else:
            raise ValueError(f'{action_choice} is not a valid strategy.')
        tree.commit_action(action)
        game.commit_action(action)
        i += 1
    winner = game.get_winner()
    if winner == 0:
        res = 0
    else:
        res = (1 if first_starts else -1) * winner
    return res


def evaluate_model_against_random(nn: NN, config: Config, action_choice='best'):
    results = [0, 0, 0]
    for i in tqdm.tqdm(range(config.test_games)):
        results[play_game_against_random(nn, config, i % 2, action_choice=action_choice)] += 1
    return results


def elo_from_prob(p: float):
    if p < 1e-6:
        return -10000
    if p > 1 - 1e-6:
        return 10000
    return -400 * np.log10(1 / p - 1)


elo_from_prob = np.vectorize(elo_from_prob)


def calculate_distribution(positive, negative):
    dist = beta(positive, negative)
    ub = positive / (positive + negative) + dist.std()
    lb = positive / (positive + negative) - dist.std()
    return elo_from_prob(lb), elo_from_prob(positive / (positive + negative)), elo_from_prob(ub)


def evaluate_models_against_random(games: int, config: Config, action_choice='best'):
    assert action_choice in action_choices
    files = glob.glob("../models/*")
    models_list: List[str] = []
    models_results: Dict[str, List[int, int, int]] = {}
    models: Dict[str, NN] = {}
    for file_path in files:
        nn = NN(config, file_path=file_path)
        models_list.append(file_path)
        models[file_path] = nn
        models_results[file_path] = [0, 0, 0]
    for name, model in models.items():
        models_results[name] = evaluate_model_against_random(model, config, action_choice=action_choice)
    sorted_models = list(
        sorted(models_results.items(), key=lambda x: x[1][-1] - x[1][1])
    )
    for name, (t, w, l) in sorted_models:
        lb, av, ub = calculate_distribution(w + t / 2, l + t / 2)
        print(f'{name:>30}: +{w}-{l}={t}, score: {(w + t / 2) / games}, elo: {lb:.0f} - {av:.0f} - {ub:.0f}')


def evaluate_models_against_nn(nn: NN, config: Config,
                               action_choice='best'):
    assert action_choice in action_choices
    print('calculating base elo...')
    base_results = evaluate_model_against_random(nn, config, action_choice=action_choice)
    base_lb, base_av, base_ub = calculate_distribution(base_results[1] + base_results[0] / 2,
                                                       base_results[-1] + base_results[0] / 2)
    files = glob.glob("../models/*")
    models_list: List[str] = []
    models_results: Dict[str, List[int, int, int]] = {}
    models: Dict[str, NN] = {}
    for file_path in files:
        nn = NN(config, file_path=file_path)
        models_list.append(file_path)
        models[file_path] = nn
        models_results[file_path] = [0, 0, 0]
    for name, model in models.items():
        models_results[name] = models_match(nn, model, config, action_choice=action_choice)
    sorted_models = list(
        sorted(models_results.items(), key=lambda x: x[1][-1] - x[1][1])
    )
    print(
        f'Base NN results: +{base_results[1]}-{base_results[-1]}={base_results[0]}, score: {(base_results[1] + base_results[1] / 2) / config.test_games}, elo: {base_lb:.0f} - {base_av:.0f} - {base_ub:.0f}')
    for name, (t, w, l) in sorted_models:
        lb, av, ub = calculate_distribution(w + t / 2, l + t / 2)
        print(f'{name:>30}: +{w}-{l}={t}, score: {(w + t / 2) / config.test_games}, elo: {lb:.0f} - {av:.0f} - {ub:.0f}')


def play_against_minmax(nn: NN, config: Config, first_starts: bool, action_choice: str = 'best', mm: MinMax = None):
    assert action_choice in action_choices
    if mm is None:
        mm = MinMax()
    game = Game().copy()
    tree = MCTS(config)
    policy = nn.policy_function
    i: int = 0
    while not game.is_terminal():
        legal_actions = game.get_actions()
        if first_starts and i % 2 == 0 or not first_starts and i % 2 == 1:
            probs, _ = tree.run(game.copy(), policy)
            if action_choice == "best":
                action = np.argmax(probs)
            elif action_choice == "probabilities":
                action = np.random.choice(Game.num_actions, p=probs)
            elif action_choice == "random":
                action = random.choice(legal_actions)
            elif action_choice == "random_sharp":
                action = random_sharp(probs, legal_actions)
            else:
                raise ValueError(f'{action_choice} is not a valid strategy')
        else:
            probs, _ = mm.policy_function(game.position)
            action = np.random.choice(Game.num_actions, p=probs)
        tree.commit_action(action)
        game.commit_action(action)
        i += 1
    winner = game.get_winner()
    if winner == 0:
        res = 0
    else:
        res = (1 if first_starts else -1) * winner
    return res


def evaluate_model_against_minmax(nn: NN, config: Config, action_choice='best', mm: MinMax = None):
    results = [0, 0, 0]
    if mm is None:
        mm = MinMax()
    for i in tqdm.tqdm(range(config.test_games)):
        results[play_against_minmax(nn, config, i % 2, action_choice=action_choice, mm=mm)] += 1
    return results


def evaluate_models_against_minmax(config: Config, action_choice='best'):
    assert action_choice in action_choices
    mm = MinMax()
    files = glob.glob("../models/*")
    models_list: List[str] = []
    models_results: Dict[str, List[int, int, int]] = {}
    models: Dict[str, NN] = {}
    for file_path in files:
        nn = NN(config, file_path=file_path)
        models_list.append(file_path)
        models[file_path] = nn
        models_results[file_path] = [0, 0, 0]
    for name, model in models.items():
        models_results[name] = evaluate_model_against_minmax(model, config,
                                                             action_choice=action_choice, mm=mm)
    sorted_models = list(
        sorted(models_results.items(), key=lambda x: x[1][-1] - x[1][1])
    )
    for name, (t, w, l) in sorted_models:
        lb, av, ub = calculate_distribution(w + t / 2, l + t / 2)
        print(f'{name:>30}: +{w}-{l}={t}, score: {(w + t / 2) / config.test_games}, elo: {lb:.0f} - {av:.0f} - {ub:.0f}')

def play_and_visualize_against_minmax(nn: NN, first_starts: bool,
        config: Config, mm: MinMax = None, action_choice: str = 'best', ):
    assert action_choice in action_choices
    if mm is None:
        mm = MinMax()
    game = Game().copy()
    tree = MCTS(config)
    policy = nn.policy_function
    i: int = 0
    while not game.is_terminal():
        legal_actions = game.get_actions()
        if first_starts and i % 2 == 0 or not first_starts and i % 2 == 1:
            probs, results = tree.run(game.copy(), policy)
            if action_choice == "best":
                action = np.argmax(probs)
            elif action_choice == "probabilities":
                action = np.random.choice(Game.num_actions, p=probs)
            elif action_choice == "random":
                action = random.choice(legal_actions)
            elif action_choice == "random_sharp":
                action = random_sharp(probs, legal_actions)
            else:
                raise ValueError(f'{action_choice} is not a valid strategy')
        else:
            probs, results = mm.policy_function(game.position)
            action = np.random.choice(Game.num_actions, p=probs)
        
        print(f"Making move {action} with probability {probs[action] * 100:.2f}%"
                f" outcome probabilities: X - {results[1]:.2f}, tie - {results[0]:.2f}, O - {results[2]:.2f}")
        print(f'{(probs.reshape((Game.board_height, Game.board_width)) * 100).round(2)}')
        tree.commit_action(action)
        game.commit_action(action)
        pos = list(game.position.visualize())
        c = pos[action // Game.board_height * (Game.board_height + 1) + action % Game.board_height]
        c = Fore.RED + c + Fore.RESET
        pos[action // Game.board_height * (Game.board_height + 1) + action % Game.board_height] = c
        print(''.join(pos))
        i += 1
    winner = game.get_winner()
    if winner == 0:
        res = 0
    else:
        res = (1 if first_starts else -1) * winner
    return res