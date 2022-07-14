import numpy as np
from numpy import ndarray
from typing import Tuple

NUM_ACTIONS: int = 9
BOARD_WIDTH = 3
BOARD_HEIGHT = 3
NUM_LAYERS = 4
IN_ROW = 3
Image = str


class Position:
    """Represents position in the game"""

    def __init__(self, board: ndarray):
        self.board: ndarray = board  # Array with shape (BOARD_HEIGHT, BOARD_WIDTH)
        assert self.board.shape == (BOARD_HEIGHT, BOARD_WIDTH)

    def to_image(self) -> str:
        """Returns the representation of the position that
        preserves all the information about the board"""
        res = ""
        for i in self.board.reshape(-1):
            res += str(int(round(i + 1)))
        return res

    def get_current_move(self) -> int:
        """Returns 1 if player who moved first should move now,
        -1 if the player who moved second should move now"""
        return 1 if (self.board != 0).sum() % 2 == 0 else -1

    def get_winner(self) -> int:
        """Returns 1 if player who moved first won,
        0 if the game is a tie,
        -1 if the player who moved second won,
        2 if the game is not over"""
        # Check for horizontal slices
        for i in range(BOARD_HEIGHT):
            for j in range(BOARD_WIDTH - IN_ROW + 1):
                board_slice = self.board[i, j: j + IN_ROW + 1]
                if (board_slice == 1).all():
                    return 1
                if (board_slice == -1).all():
                    return -1
        # Check for vertical slices
        for i in range(BOARD_HEIGHT - IN_ROW + 1):
            for j in range(BOARD_WIDTH):
                board_slice = self.board[i: i + IN_ROW, j]
                if (board_slice == 1).all():
                    return 1
                if (board_slice == -1).all():
                    return -1
        # Check for diagonal slices
        for i in range(BOARD_HEIGHT - IN_ROW + 1):
            for j in range(BOARD_WIDTH - IN_ROW + 1):
                board_slice = self.board[i: i + IN_ROW, j: j + IN_ROW].diagonal()
                if (board_slice == 1).all():
                    return 1
                if (board_slice == -1).all():
                    return -1
        flipped = np.fliplr(self.board)
        for i in range(BOARD_HEIGHT - IN_ROW + 1):
            for j in range(BOARD_WIDTH - IN_ROW + 1):
                board_slice = flipped[i: i + IN_ROW, j: j + IN_ROW].diagonal()
                if (board_slice == 1).all():
                    return 1
                if (board_slice == -1).all():
                    return -1
        return 0 if (self.board != 0).all() else 2

    def vectorize(self) -> np.ndarray:
        """Returns normalized vector that represents the position for NN training.
        The shape of the state is (4, 3, 3)"""
        result = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, NUM_LAYERS))
        for (i, j), v in np.ndenumerate(self.board):
            if v == 1:
                result[i, j, 0] = 1
            elif v == -1:
                result[i, j, 1] = 1
        result[:, :, 2] = self.get_current_move()
        result[:, :, 3] = np.array(self.board).reshape((BOARD_HEIGHT, BOARD_WIDTH))
        return result

    def copy(self):
        return Position(self.board.copy())

    def visualize(self) -> str:
        def symbol_from_int(x):
            return {-1: "O", 0: ".", 1: "X"}[x]

        b = np.vectorize(symbol_from_int)(self.board)
        return "\n".join(["".join(x) for x in b])


def position_from_image(pos: Image) -> Position:
    """Returns the position from the image of the position"""
    return Position(
        np.array([int(x) - 1 for x in pos]).reshape(BOARD_HEIGHT, BOARD_WIDTH)
    )


START_POSITION: Position = Position(np.zeros((BOARD_HEIGHT, BOARD_WIDTH)))


class Game:
    """Describes the logic of the game and provides appropriate interface"""

    num_actions: int = NUM_ACTIONS  # Number of actions possible in the game
    board_height = BOARD_HEIGHT
    board_width = BOARD_WIDTH
    num_layers = NUM_LAYERS

    def __init__(self, position=START_POSITION):
        self.position: Position = position.copy()  # Current in-game position

    def is_terminal(self) -> bool:
        """Returns true if the position of the game is terminal"""
        return self.position.get_winner() != 2

    def get_winner(self) -> int:
        """Returns 1 if player who moved first won, 0 if the game is a tie,
        -1 if the player who moved second won"""
        return self.position.get_winner()

    def get_actions(self) -> ndarray:
        """Returns the list of actions that are possible from the current position"""
        return np.array(
            np.where(self.position.board.reshape(Game.num_actions) == 0)
        ).reshape(-1)

    def get_current_move(self) -> int:
        """Returns 1 if player who moved first should move now,
        -1 if the player who moved second should move now"""
        return self.position.get_current_move()

    def copy(self):
        """Returns the copy of the game"""
        new_game = Game()
        new_game.num_actions = self.num_actions
        new_game.position = self.position.copy()
        return new_game

    def commit_action(self, action: int):
        """Makes a move according to the id of the action"""
        if self.position.board.reshape(-1)[action] != 0:
            raise IndexError(f"The {action}-th cell is already taken")
        self.position.board.reshape(-1)[action] = self.get_current_move()


def augment_data(
        x: ndarray, y_act: ndarray, y_val: ndarray
) -> Tuple[ndarray, ndarray, ndarray]:
    batch_size = x.shape[0]
    y_act = y_act.reshape((-1, 3, 3))
    result_x: ndarray = np.zeros((batch_size * 4, 3, 3, 4))
    result_y: ndarray = np.zeros((batch_size * 4, 3, 3))
    result_y_val: ndarray = np.zeros((batch_size * 4, 3))

    result_x[:batch_size] = x
    result_y[:batch_size] = y_act
    result_y_val[:batch_size] = y_val

    result_x[batch_size: batch_size * 2] = np.rot90(x, k=1, axes=(1, 2))
    result_y[batch_size: batch_size * 2] = np.rot90(y_act, k=1, axes=(1, 2))
    result_y_val[batch_size: batch_size * 2] = y_val

    result_x[batch_size * 2: batch_size * 3] = np.rot90(x, k=2, axes=(1, 2))
    result_y[batch_size * 2: batch_size * 3] = np.rot90(y_act, k=2, axes=(1, 2))
    result_y_val[batch_size * 2: batch_size * 3] = y_val

    result_x[batch_size * 3:] = np.rot90(x, k=3, axes=(1, 2))
    result_y[batch_size * 3:] = np.rot90(y_act, k=3, axes=(1, 2))
    result_y_val[batch_size * 3:] = y_val

    return result_x, result_y.reshape((-1, 9)), result_y_val


def test_position():
    pos = Position(np.array([[0, 0, 1], [0, 1, -1], [1, -1, 0], ]))
    assert pos.get_winner() == 1
    assert pos.get_current_move() == -1
    assert pos.to_image() == "112120201"
    assert (position_from_image(pos.to_image()).board == pos.board).all()
    pos = Position(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], ]))
    assert pos.get_winner() == 2
    assert pos.get_current_move() == 1
    pos.vectorize()
    pos2 = pos.copy()
    pos2.board[0, 0] = 1
    assert not (pos.board == pos2.board).all()
    pos = Position(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], ]))
    assert pos.get_winner() == 1


def test_game():
    game = Game()
    assert not game.is_terminal()
    assert game.get_winner() == 2
    print(game.get_actions().shape)
    print((Game.num_actions,))
    assert game.get_actions().shape == (Game.num_actions,)
    assert (game.get_actions() == np.arange(game.num_actions)).all()
    game.commit_action(0)
    assert not game.is_terminal()
    assert (game.get_actions() == np.arange(1, game.num_actions)).all()
    game.commit_action(1)
    game.commit_action(4)
    game.commit_action(2)
    game.commit_action(8)
    assert game.get_winner() == 1
    assert game.is_terminal()
