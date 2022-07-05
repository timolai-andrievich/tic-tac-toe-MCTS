from Game import NUM_ACTIONS, Game, Position, position_from_image
from NN import NN, PolicyNN, ValueNN
from MCTS import Node, MCST
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
    assert g.get_current_move() == -1
    assert g.get_actions() == [4, 5, 7, 8]
    assert g.is_terminal() == True


def test_nnmodel():
    pos = Position([1, -1, -1, 1, 0, 0, 1, 0, 0])
    state = pos.vectorize()[..., np.newaxis]
    ValueNN()(state)
    PolicyNN()(state)


def test_nn():
    nn = NN()
    pos = Position([1, -1, -1, 1, 0, 0, 1, 0, 0])
    state = pos.vectorize()
    nn.policy_function(pos)
    batch = [
        (pos.to_image(), (np.zeros(NUM_ACTIONS), 0,),),
        (pos.to_image(), (np.zeros(NUM_ACTIONS), 0,),),
    ]
    nn.train(batch)


def test_node():
    root = Node(None, 0, 1)
    game = Game()
    probs = np.array([1] * NUM_ACTIONS) / NUM_ACTIONS
    assert root.is_leaf()
    assert root.is_root()
    root.expand(game, probs)
    _: Tuple[int, Node] = root.select()
    action: int = _[0]
    node: Node = _[1]
    node.update_recursive(np.array([0, 1, 0]))
    assert (root.results == np.array([0, 1, 0])).all()
    assert (node.results == np.array([0, 1, 0])).all()


def test_tree():
    nn = NN()
    game = Game()
    tree = MCST()
    tree.run(game, nn.policy_function, 100)


if __name__ == "__main__":
    test_nn()
