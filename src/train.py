from re import A
from Game import Game, Position, NUM_ACTIONS, Image, START_POSITION
from NN import NN
from MCTS import MCST, Node
from typing import List, Tuple, Dict
import random

BATCH_SIZE = 200
GAME_COUNT = 100
ITERATION_COUNT = 10000


def pick_action(probs: Dict[int, float]) -> int:
    """Analyzes the list of actions with probabilities and returns the most appropriate one"""


def make_target(
    probs: Dict[Image, Dict[int, float]], scores: List[float]
) -> List[Tuple[Image, Tuple[Dict[int, float], float]]]:
    """Compiles the probabilities of functions and final evaluation scores into one list and returns it"""


def generate_game(network: NN) -> List[Tuple[Image, Tuple[Dict[int, float], float]]]:
    game = Game()
    mcst = MCST(game, network.policy_function)
    probabilities = {}
    while not game.is_terminal():
        probs = mcst.run(game)
        probabilities[game._position.to_image] = probs
        action = pick_action(probs)
        game.commit_action(action)
        mcst.commit_action(action)
    game.assign_scores()
    scores = game.get_scores()
    return make_target(probabilities, scores)


def create_batch(network: NN) -> List[Tuple[Image, Tuple[Dict[int, float], float]]]:
    """Creates the batch of positions for training"""


def train_iteration(network: NN):
    batch = create_batch()
    NN.train(batch)


def train(network: NN):
    for i in range(ITERATION_COUNT):
        train_iteration(network)
