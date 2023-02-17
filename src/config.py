"""Describes configuration class, containing parameters
for neural networks and Monte-Carlo Tree Search.
"""


class Config:  # pylint: disable=too-few-public-methods
    """Class that contains hyperparameters for the neural network and training process
    """
    # Training-related parameters
    iteration_count = 12
    checkpoints_interval = 10000
    games_in_iteration = 25
    batch_size = 64
    buffer_size = 1000
    temp = 1
    learning_rate = 1e-3
    epochs = 10
    # Exploration noise
    starting_exploration_noise = 1
    exploration_decay = 0.90
    min_exploration_noise = 0.15
    exploration_noise = starting_exploration_noise

    # MCTS Parameters
    mcts_playout = 80
    c_impact = 5

    # Game parameters
    max_moves = 9
