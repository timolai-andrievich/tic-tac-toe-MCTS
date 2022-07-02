import cProfile
import glob
import itertools
from os import stat
from turtle import pos
import numpy as np
from torch import rand
from Game import NUM_ACTIONS, Game, Image, START_POSITION, Position
from NN import NN
from MCTS import MCST
from typing import Dict, List, Tuple
from numpy import ndarray
import tqdm
import random

iteration_count = 1
games_in_iteration = 50
mcts_playout = 50
batch_size = -1
checkpoints = 10
test_cp = 10
random_playout = 6000
test_games = 100
intermediate_test_games = 10
exploration_noise = .2


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
    nn = NN(file=file_name)
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
                probs = probs * (1 - exploration_noise) + exploration_noise * np.random.dirichlet(np.ones(NUM_ACTIONS))
                legal_actions = game.get_actions()
                for a in range(NUM_ACTIONS):
                    if not a in legal_actions:
                        probs[a] = 0
                probs = probs / probs.sum()
                action = np.random.choice(NUM_ACTIONS, p=probs)
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


def play_against_random_nn(nn: NN, nn_starts: bool) -> int:
    """Returns 1 if nn wins, 0 if the game is tied, -1 if pure mcts wins"""
    game = Game().copy()
    random_nn = NN()
    tree_nn = MCST(game.copy(), nn.policy_function, mcts_playout)
    tree_pure = MCST(game.copy(), random_nn.policy_function, random_playout)
    if nn_starts:
        trees = [tree_nn, tree_pure]
        policies = [nn.policy_function, random_nn.policy_function]
    else:
        trees = [tree_pure, tree_nn]
        policies = [random_nn.policy_function, nn.policy_function]
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


def test(nn: NN, silent=False):
    games_results = [0] * 3
    for i in tqdm.tqdm(range(test_games)):
        winner = play_against_random_nn(nn, i % 2 == 1)
        games_results[winner] += 1
    if not silent:
        print(f"""Games won: {games_results[1]}
Games tied: {games_results[0]}
Games lost: {games_results[-1]}
Win/tie rate: {(games_results[0] + games_results[1]) / test_games * 100:.2f}%""")
    return games_results


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

def models_test():
    models = []
    for model in glob.glob('../models/*'):
        nn = NN(file=model)
        t, w, l = test(nn, silent=True)
        models.append((l, t, w, model))
    models.sort()
    print("Top ten models who swore:")
    for t, w, l, model in models:
        print(f'+{w:<3}-{l:<3}={t:<3} {model}')


def models_play(nn1: NN, nn2: NN, first_starts: bool):
    """Returns 1 if the first nn wins, 0 if the game is tied, -1 if thesecond nn wins"""
    game = Game().copy()
    tree1 = MCST(game.copy(), nn1.policy_function, mcts_playout)
    tree2 = MCST(game.copy(), nn2.policy_function, mcts_playout)
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
        for a in range(NUM_ACTIONS):
            if not a in legal_actions:
                probs[a] = 0
        probs = probs / probs.sum()
        # action = np.random.choice(NUM_ACTIONS, p=probs)
        action = np.argmax(probs)
        trees[0].commit_action(action)
        trees[1].commit_action(action)
        game.commit_action(action)
    winner = game.get_winner()
    if winner == 0:
        res =  0
    else:
        res =  (1 if first_starts else -1) * winner
    return res


def models_match(nn1: NN, nn2: NN, games:int = test_games):
    res = [0, 0, 0]
    for i in range(games):
        winner = models_play(nn1, nn2, i % 2 == 0)
        res[winner] += 1
    return res


def models_tournament_gauntlet():
    i: int = 0
    files = glob.glob('../models/*')
    models_list: Tuple[NN, str] = []
    models_results: Dict[str, List[int, int, int]] = {} 
    for file in files:
        nn = NN(file=file)
        models_list.append([nn, file])
        models_results[file] = [0, 0, 0]
    models_list.append([NN(), 'random'])
    models_results['random'] = [0, 0, 0]
    while len(models_list) > 1:
        print(f'Models left: {len(models_list)}')
        new_models_list = []
        random.shuffle(models_list)
        for i in range(0, len(models_list), 2):
            models = models_list[i:i+2]
            if len(models) == 1:
                (nn, file), = models
                new_models_list.append((nn, file))
                continue
            (nn1, file1), (nn2, file2) = models
            res = models_match(nn1, nn2)
            t, w, l = res
            print(f'{file1} - {file2}: +{w}-{l}={t}')
            models_results[file1][0] += res[0]
            models_results[file1][1] += res[1]
            models_results[file1][-1] += res[-1]

            models_results[file2][0] += res[0]
            models_results[file2][1] += res[-1]
            models_results[file2][-1] += res[1]
            
            if l > w:
                new_models_list.append((nn2, file2))
            else:
                new_models_list.append((nn1, file1))
        models_list = new_models_list
    (_, file), = models_list
    sorted_models = sorted(models_results.items(), key = lambda x: x[1][-1] - x[1][1])
    for model, (t, w, l) in sorted_models:
        print(f'{model}: +{w}-{l}={t}')


def models_tournament_round():
    files = glob.glob('../models/*')
    models_list: List[str] = []
    models_results: Dict[str, List[int, int, int]] = {} 
    models: Dict[str, NN] = {}
    for file in files:
        nn = NN(file=file)
        models_list.append(file)
        models[file] = nn
        models_results[file] = [0, 0, 0]
    models_list.append('random')
    models['random'] = NN()
    models_results['random'] = [0, 0, 0]
    matchups = itertools.combinations(models_list, 2)
    for file1, file2 in tqdm.tqdm(list(matchups)):
        nn1 = models[file1]
        nn2 = models[file2]
        res = models_match(nn1, nn2)
        t, w, l = res
        print(f'{file1} - {file2}: +{w}-{l}={t}')
        models_results[file1][0] += res[0]
        models_results[file1][1] += res[1]
        models_results[file1][-1] += res[-1]

        models_results[file2][0] += res[0]
        models_results[file2][1] += res[-1]
        models_results[file2][-1] += res[1]
    sorted_models = sorted(models_results.items(), key = lambda x: x[1][-1] - x[1][1])
    for model, (t, w, l) in sorted_models:
        print(f'{model}: +{w}-{l}={t}')

def profile():
    import pstats
    with cProfile.Profile() as p:
        train()
    stats = pstats.Stats(p)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()

def show_moves(nn: NN):
    positions = [
        [
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        ],
        [
            0, 0, 0,
            0, 1, 0,
            0, 0, 0,
        ],
        [
            0, 0, 0,
            -1, 1, 0,
            0, 0, 0,
        ],
    ]
    positions = [Position(p) for p in positions]
    for position in positions:
        probs, eval = nn.policy_function(position)
        l = position.get_actions()
        for i in range(NUM_ACTIONS):
            if not i in l:
                probs[i] = 0
            probs /= probs.sum()
        action = np.argmax(probs)
        new_position = position.with_move(action)
        print(f'{position.visualize()}\n{probs}\n{new_position.visualize()}\n\n')


def main():
    profile()
    train()

if __name__ == "__main__":
    main()
