"""Contains utility functions and classes, related to
playing matches and tournaments between game strategies.
"""
import glob
from typing import Dict, List, Tuple

import numpy as np
import tqdm
from scipy.stats import beta

from .game import Game
from .policy import Model
from .config import Config
from .player import MctsPlayer, ModelPlayer, Player


def play_game(player_1: Player, player_2: Player) -> int:
    """Simulates a game between two players.

    Args:
        player_1 (Player)
        player_2 (Player)

    Returns:
        int: Result of the game: 1 if player_1 wins,
        0 if the game ended in a tie, -1 if player_2 wins.
    """
    game = Game().copy()
    players = player_1, player_2
    move = 0
    while not game.is_finished():
        # Alternates between the players
        action = players[move % 2].get_action(game.copy())
        game.commit_action(action)
        move += 1
    return game.get_winner()


def play_match(player_1: Player,
               player_2: Player,
               games: int,
               silent=False) -> List[int]:
    """Simulates a match between two players.

    Args:
        player_1 (Player)
        player_2 (Player)
        games (int): Number of games in a match.
        silent (bool, optional): If True, shows match progress bar. Defaults to False.

    Returns:
        List[int]: Results of the game in format [ties, player_1 wins, player_2 wins].
    """
    results = [0, 0, 0]
    for i in range(games) if silent else tqdm.tqdm(range(games)):
        result = play_game(player_1, player_2)
        if i % 2:
            result = -result
        results[result] += 1
        player_1, player_2 = player_2, player_1
    return results


def elo_from_expected_score(expected_score: np.ndarray) -> np.ndarray:
    """Calculates relative elo rating from expecred score
    in a game in a match between two players.

    Args:
        expected_score (float): Expected score of a game between two players.

    Returns:
        float: Relative elo rating.
    """
    expected_score = np.clip(expected_score, 1e-20, 1 - 1e-20)
    return -400 * np.log10(1 / expected_score - 1)


def calculate_distribution(
        positive: float,
        negative: float,
        probability: float = .95) -> Tuple[float, float, float]:
    """Calculates lower and upper bounds for elo rating using beta distribution.
    Bounds are chosen such that the probability of elo being within the returned
    range is `probability`.

    Args:
        positive (float): Number of positive outcomes (such as wins).
        negative (float): Number of negative outcomes (such as losses).
        probability (float): The probability of true elo being within returned range.

    Returns:
        Tuple[float, float, float]: (Lower rating bound, Expected rating, Upper rating bound).
    """
    assert 0 < probability < 1
    dist = beta(positive, negative)

    def binsearch(target: float, eps: float = 1e-6):
        low, high = 0, 1
        while high - low > eps:
            middle = (high + low) / 2
            if dist.cdf(middle) > target:
                high = middle
            else:
                low = middle
        return (high + low) / 2

    upper_bound = binsearch((1 + probability) / 2)
    lower_bound = binsearch((1 - probability) / 2)
    return (
        elo_from_expected_score(lower_bound),
        elo_from_expected_score(positive / (positive + negative)),
        elo_from_expected_score(upper_bound),
    )


def print_results_table(table: Dict[str, Tuple[int, int, int]]):
    """Prints the results table to the standart output.

    Args:
        table (Dict[str, Tuple[int, int, int]]): Results table,
        represented as a dictionary in (name / ties, wins, losses) format.
    """
    for name, (ties, wins, loses) in sorted(table.items(),
                                            key=lambda x: x[1][-1] - x[1][1]):
        lower_bound, expected_rating, upper_bound = (calculate_distribution(
            wins + ties / 2, loses + ties / 2))
        print(
            f"{name:>30}: {f'+{wins}-{loses}={ties}':>15}, "
            f"elo: {lower_bound:^6.0f} - {expected_rating:^6.0f} - {upper_bound:^6.0f}"
        )


def evaluate_models_against_player(config: Config,
                                   player: Player,
                                   games: int,
                                   path="./models"):
    """Loads models from files in `path` folder, and evaluates
    all of them against a given player using MCTS.
    Prints results of evaluation to stdout.

    Args:
        config (Config): MCTS and NN parameters.
        player (Player): Player to evaluate against.
        games (int): Games to play in one match.
        path (str, optional): Path to folder with model weights. Defaults to "./models".
    """
    models_results: Dict[str, List[int, int, int]] = {}
    for file_path in glob.glob(f"{path}/*"):
        model = Model(config, file_path=file_path)
        model_player = MctsPlayer(model, config)
        models_results[file_path] = [0, 0, 0]
        models_results[file_path] = play_match(model_player, player, games)
    print_results_table(models_results)
    return models_results


def evaluate_pure_models_against_player(config: Config,
                                        player: Player,
                                        games: int,
                                        path="./models"):
    """Loads models from files in `path` folder, and evaluates
    all of them against a given player.
    Prints results of evaluation to stdout.

    Args:
        config (Config): MCTS and NN parameters.
        player (Player): Player to evaluate against.
        games (int): Games to play in one match.
        path (str, optional): Path to folder with model weights. Defaults to "./models".
    """
    models_results: Dict[str, List[int, int, int]] = {}
    for file_path in glob.glob(f"{path}/*"):
        model = Model(config, file_path=file_path)
        model_player = ModelPlayer(model)
        models_results[file_path] = [0, 0, 0]
        models_results[file_path] = play_match(model_player, player, games)
    print_results_table(models_results)
    return models_results
