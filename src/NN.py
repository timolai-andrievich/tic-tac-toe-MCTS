from Game import Position, Image
from typing import Tuple, Dict, List


class NN:
    """Provides the interface for an neural network model"""

    def __init__(self):
        pass

    def policy_function(self, position: Position) -> Tuple[Dict[int, float], float]:
        """Evaluates the position and returns probabilities of actions and evaluation score"""

    def train(self, batch: List[Tuple[Image, Tuple[Dict[int, float], float]]]):
        """Trains the NN on a batch of data collected from self-play"""
