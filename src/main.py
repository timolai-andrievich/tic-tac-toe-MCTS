import numpy as np
from torch import rand
from Game import Game, Image, START_POSITION, Position
from NN import NN
from MCTS import MCST
from typing import List, Tuple
from numpy import ndarray
import tqdm
import random

iteration_count = 5000
games_in_iteration = 20
mcts_playout = 400
batch_size = -1
checkpoints = 300
test_cp = 50
pure_playout = 6000
test_games = 1000
intermediate_test_games = 6


def make_batch(
    training_data: List[Tuple[Image, Tuple[ndarray, float]]],
    batch_size: int = batch_size,
) -> List[Tuple[Image, Tuple[ndarray, float]]]:
    if batch_size == -1:
        return training_data
    batch: List[Tuple[Image, Tuple[ndarray, float]]] = []
    for _ in range(batch_size):
        batch.append(random.choice(training_data))
    return batch


def train(file_name=None):
    nn = NN(file=file_name, use_gpu=True)
    for i in tqdm.tqdm(range(iteration_count)):
        training_data: List[Tuple[Image, Tuple[ndarray, float]]] = []
        for _ in range(games_in_iteration):
            game = Game().copy()
            debug_x = game.is_terminal()
            if debug_x:
                pass
            tree = MCST(game.copy(), nn.policy_function, mcts_playout)
            while not game.is_terminal():
                probs, eval = tree.run(game.copy(), nn.policy_function)
                action, _ = tree._root.select()
                training_data.append((game._position.to_image(), (probs, eval)))
                game.commit_action(action)
                tree.commit_action(action)
        batch = make_batch(training_data)
        nn.train(batch)
        if i % checkpoints == 0:
            nn.dump(info=f'iteration_{i}')
        if i % test_cp == 0:
            intermediate_test(nn)
    nn.dump()
    return nn


def random_policy(position: Position) -> Tuple[ndarray, float]:
    return np.array([1 / 9] * 9), 0


def play_against_pure_mcts(nn: NN, nn_starts: bool) -> int:
    """Returns 1 if nn wins, 0 if the game is tied, -1 if pure mcts wins"""
    game = Game().copy()
    tree_nn = MCST(game.copy(), nn.policy_function, mcts_playout)
    tree_pure = MCST(game.copy(), random_policy, pure_playout)
    if nn_starts:
        trees = [tree_nn, tree_pure]
        policies = [nn.policy_function, random_policy]
    else:
        trees = [tree_pure, tree_nn]
        policies = [random_policy, nn.policy_function]
    i: int = 0
    while not game.is_terminal():
        trees[i & 1].run(game.copy(), policies[i & 1])
        action, _ = trees[i & 1]._root.select()
        trees[0].commit_action(action)
        trees[1].commit_action(action)
        game.commit_action(action)
    winner = game.get_winner()
    if winner == 0:
        res =  0
    else:
        res =  (1 if nn_starts else -1) * winner
    print(f'nn starts: {nn_starts}, winner: {winner}, res: {res}')
    return res



def test_mcts():
    nn = NN()
    game = Game()
    tree = MCST(game.copy(), nn.policy_function, 500)
    print(tree.run(game.copy(), nn.policy_function))
    game.commit_action(4)
    tree.commit_action(4)
    game.commit_action(3)
    tree.commit_action(3)
    print(tree.run(game.copy(), nn.policy_function))
    game.commit_action(0)
    tree.commit_action(0)
    print(tree.run(game.copy(), nn.policy_function))


def test(nn: NN):
    print(f'Should be 0: {nn.policy_function(START_POSITION)[1]}')
    print(f'Should be 1: {nn.policy_function(Position([0, 0, 0,-1, 1, 0, 0, 0, 0]))[1]}')
    print(f'Should be 1, but more obvious: {nn.policy_function(Position([1, 1, 0,-1, 0, 0,-1, 0, 0]))[1]}')
    print(f'Should be -1, obvious: {nn.policy_function(Position([-1,-1, 0, 1, 0, 0, 1, 0, 0]))[1]}')
    game = Game().copy()
    tree = MCST(game.copy(), nn.policy_function, mcts_playout)
    print(f'Should be 0, MCTS: {tree.run(game.copy(), nn.policy_function)[1]}')
    game.commit_action(4)
    tree.commit_action(4)
    game.commit_action(3)
    tree.commit_action(3)
    print(f'Should be 1, MCTS: {tree.run(game.copy(), nn.policy_function)[1]}')
    game.commit_action(0)
    tree.commit_action(0)
    print(f'Should be 1, MCTS: {tree.run(game.copy(), nn.policy_function)[1]}')
    games_results = [0] * 3
    for i in tqdm.tqdm(range(test_games)):
        winner = play_against_pure_mcts(nn, i % 2 == 1)
        games_results[winner] += 1
    print(f"""Games won: {games_results[1]}
Games tied: {games_results[0]}
Games lost: {games_results[-1]}
Win/tie rate: {(games_results[0] + games_results[1]) / test_games * 100:.2f}%""")


def intermediate_test(nn: NN):
    print(f'Should be 0: {nn.policy_function(START_POSITION)[1]}')
    print(f'Should be 1: {nn.policy_function(Position([0, 0, 0,-1, 1, 0, 0, 0, 0]))[1]}')
    print(f'Should be 1, but more obvious: {nn.policy_function(Position([1, 1, 0,-1, 0, 0,-1, 0, 0]))[1]}')
    print(f'Should be -1, obvious: {nn.policy_function(Position([-1,-1, 0, 1, 0, 0, 1, 0, 0]))[1]}')
    game = Game().copy()
    tree = MCST(game.copy(), nn.policy_function, mcts_playout)
    print(f'Should be 0, MCTS: {tree.run(game.copy(), nn.policy_function)[1]}')
    game.commit_action(4)
    tree.commit_action(4)
    game.commit_action(3)
    tree.commit_action(3)
    print(f'Should be 1, MCTS: {tree.run(game.copy(), nn.policy_function)[1]}')
    game.commit_action(0)
    tree.commit_action(0)
    print(f'Should be 1, MCTS: {tree.run(game.copy(), nn.policy_function)[1]}')
    games_results = [0] * 3
    print("Playing intermediate tests games:")
    for i in tqdm.tqdm(range(intermediate_test_games)):
        winner = play_against_pure_mcts(nn, i % 2 == 1)
        games_results[winner] += 1
    print(f"""Games won: {games_results[1]}
Games tied: {games_results[0]}
Games lost: {games_results[-1]}
Win/tie rate: {(games_results[0] + games_results[1]) / intermediate_test_games * 100:.2f}%""")


def main():
    nn = train()
    test(nn)


if __name__ == "__main__":
    main()
