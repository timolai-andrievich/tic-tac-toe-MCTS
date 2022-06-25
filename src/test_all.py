from Game import NUM_ACTIONS, Game, Position, position_from_image
from NN import NN, NNModel
import torch
import numpy as np


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
