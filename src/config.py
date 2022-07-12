class Config:
    iteration_count = 100
    games_in_iteration = 50
    mcts_playout = 400
    batch_size = 128
    checkpoints = 10
    test_games = 50
    starting_exploration_noise = 1
    exploration_decay = 0.90
    min_exploration_noise = 0.15
    c_impact = 5
    minibatch_size = 128
    learning_rate = 1e-1
