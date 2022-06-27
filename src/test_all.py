from Game import NUM_ACTIONS, Game, Position, position_from_image
from NN import NN, NNModel
from MCTS import Node, MCST
import torch
import numpy as np
from typing import Tuple


def test_position():
    pos = Position([1, -1, -1, 1, 0, 0, 1, 0, 0])
    assert pos.get_winner() == 1
    assert pos.get_current_move() == -1
    assert pos.to_image() == "200211211"
    assert position_from_image(pos.to_image()).board == pos.board
    pos = Position([-1, -1, -1, 1, 0, 0, 1, 1, 0])
    assert pos.get_winner() == -1
    assert pos.get_current_move() == 1


def test_game():
    g = Game()
    g.commit_action(0)
    g.commit_action(1)
    g.commit_action(3)
    g.commit_action(2)
    g.commit_action(6)
    g.assign_scores()
    assert g.get_current_move() == -1
    assert g.get_actions() == [4, 5, 7, 8]
    assert g.is_terminal() == True
    assert g.get_scores() == [1, 1, 1, 1, 1]


def test_nnmodel():
    model = NNModel()
    pos = Position([1, -1, -1, 1, 0, 0, 1, 0, 0])
    pos_tensor = torch.from_numpy(pos.vectorize()).float()
    model.forward(pos_tensor)


def test_nn():
    nn = NN()
    pos = Position([1, -1, -1, 1, 0, 0, 1, 0, 0])
    pos_tensor = torch.from_numpy(pos.vectorize()).float()
    nn.policy_function(pos)
    batch = [
        (
            pos.to_image(),
            (
                np.zeros(NUM_ACTIONS),
                0,
            ),
        ),
        (
            pos.to_image(),
            (
                np.zeros(NUM_ACTIONS),
                0,
            ),
        ),
    ]
    nn.train(batch)

def test_node():
    root = Node(None, 0)
    game = Game()
    probs = np.array([1] * NUM_ACTIONS) / NUM_ACTIONS
    assert(root.is_leaf())
    assert(root.is_root())
    root.expand(game, probs)
    _: Tuple[int, Node] = root.select()
    action: int = _[0]
    node: Node = _[1]
    node.update_recursive(1)
    assert(root._avg == -1)
    assert(node._avg == 1)

def test_tree():
    nn = NN()
    game = Game()
    tree = MCST(game, nn.policy_function)
    tree.run(game, nn.policy_function)

    
