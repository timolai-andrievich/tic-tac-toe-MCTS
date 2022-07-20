import glob
from typing import Dict, List, Tuple

import numpy as np
import tqdm
from scipy.stats import beta

from game import Game
from nn import NN
from config import Config
from player import MctsPlayer, ModelPlayer, Player


def play_game(p1: Player, p2: Player) -> int:
    game = Game().copy()
    players = p1, p2
    i = 0
    while not game.is_terminal():
        action = players[i & 1].get_action(game.copy())
        game.commit_action(action)
        i += 1
    return game.get_winner()


def play_match(p1: Player, p2: Player, games: int, silent=False):
    results = [0, 0, 0]
    for i in range(games) if silent else tqdm.tqdm(range(games)):
        players = (p2, p1) if i & 1 else (p1, p2)
        result = play_game(*players) * (-1 if i & 1 else 1)
        results[result] += 1
    return results


def elo_from_prob(p: float):
    if p < 1e-6:
        return -10000
    if p > 1 - 1e-6:
        return 10000
    return -400 * np.log10(1 / p - 1)


elo_from_prob = np.vectorize(elo_from_prob)


def calculate_distribution(positive, negative) -> Tuple[float, float, float]:
    dist = beta(positive, negative)
    ub = positive / (positive + negative) + dist.std()
    lb = positive / (positive + negative) - dist.std()
    return (
        elo_from_prob(lb),
        elo_from_prob(positive / (positive + negative)),
        elo_from_prob(ub),
    )


def evaluate_models_against_player(config: Config,
                                   player: Player,
                                   games: int,
                                   path="../models"):
    files = glob.glob(f"{path}/*")
    models_results: Dict[str, List[int, int, int]] = {}
    for file_path in files:
        nn = NN(config, file_path=file_path)
        model_player = MctsPlayer(nn, config)
        models_results[file_path] = [0, 0, 0]
        models_results[file_path] = play_match(model_player, player, games)
    sorted_models = list(
        sorted(models_results.items(), key=lambda x: x[1][-1] - x[1][1]))
    for name, (t, w, l) in sorted_models:
        lb, av, ub = calculate_distribution(w + t / 2, l + t / 2)
        print(f"{name:>30}: +{w}-{l}={t}, elo: {lb:.0f} - {av:.0f} - {ub:.0f}")


def evaluate_pure_models_against_player(config: Config,
                                        player: Player,
                                        games: int,
                                        path="../models"):
    files = glob.glob(f"{path}/*")
    models_results: Dict[str, List[int, int, int]] = {}
    for file_path in files:
        nn = NN(config, file_path=file_path)
        model_player = ModelPlayer(nn)
        models_results[file_path] = [0, 0, 0]
        models_results[file_path] = play_match(model_player, player, games)
    sorted_models = list(
        sorted(models_results.items(), key=lambda x: x[1][-1] - x[1][1]))
    for name, (t, w, l) in sorted_models:
        lb, av, ub = calculate_distribution(w + t / 2, l + t / 2)
        print(f"{name:>30}: +{w}-{l}={t}, elo: {lb:.0f} - {av:.0f} - {ub:.0f}")
