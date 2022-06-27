from torch import rand
from Game import Game, Image, START_POSITION, Position
from NN import NN
from MCTS import MCST
from typing import List, Tuple
from numpy import ndarray
import tqdm
import random

iteration_count = 10000
games_in_iteration = 10
mcts_playout = 100
batch_size = -1
checkpoints = 300
test_cp = 20


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
            test(nn)
    nn.dump()
    return nn


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


def main():
    #test_mcts()
    nn = train()
    test(nn)


if __name__ == "__main__":
    main()
