"""Describes configuration class, containing parameters
for neural networks and Monte-Carlo Tree Search.
"""


class Config:  # pylint: disable=too-few-public-methods
    """Class that contains hyperparameters for the neural network
    """
    # Training-related parameters
    iteration_count = 100
    checkpoints = 10
    test_checkpoints = 1
    games_in_iteration = 50
    batch_size = 128
    test_games = 50
    temp = 1
    learning_rate = 1e-1
    minibatch_size = 128
    # Exploration noise
    starting_exploration_noise = 1
    exploration_decay = 0.90
    min_exploration_noise = 0.15
    exploration_noise = starting_exploration_noise

    # MCTS Parameters
    mcts_playout = 400
    c_impact = 5

    # Game parameters
    max_moves = 9
