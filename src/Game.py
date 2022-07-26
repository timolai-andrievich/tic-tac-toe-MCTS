"""Contains classes related to game logic and game-specific constants
"""
from typing import Tuple

import numpy as np
from numpy import ndarray

NUM_ACTIONS = 9
BOARD_WIDTH = 3
BOARD_HEIGHT = 3
NUM_LAYERS = 4
IN_ROW = 3  # Number of symbols placed in a row required to win
Image = str


class Position:
    """Represents game position
    """

    def __init__(self, board: ndarray):
        self.board: ndarray = board  # Array with shape (BOARD_HEIGHT, BOARD_WIDTH)
        assert self.board.shape == (BOARD_HEIGHT, BOARD_WIDTH)

    def to_image(self) -> str:
        """Converts position into a string, preserving all information.
        Should be reversible.

        Returns:
            str: String representation of the position.
        """
        res = ""
        for i in self.board.reshape(-1):
            res += str(int(round(i + 1)))
        return res

    def get_current_move(self) -> int:
        """Returns the integer representing the player that is to move next:
        1 corresponds to the player that moved first, -1 to the player that moved second.

        Returns:
            int
        """
        return 1 if (self.board != 0).sum() % 2 == 0 else -1

    def get_winner(self) -> int: # pylint: disable=too-many-branches
        """Returns the integer according to the result of the game.
        1 if the first pleyer won,
        0 if the game is tied,
        -1 if the second player won,
        2 if the game is not finished.

        Returns:
            int: The result of the game.
        """
        res = 2
        # Check horizontal slices
        for i in range(BOARD_HEIGHT):
            for j in range(BOARD_WIDTH - IN_ROW + 1):
                board_slice = self.board[i, j:j + IN_ROW]
                if (board_slice == 1).all():
                    res = 1
                elif (board_slice == -1).all():
                    res = -1
        # Check vertical slices
        for i in range(BOARD_HEIGHT - IN_ROW + 1):
            for j in range(BOARD_WIDTH):
                board_slice = self.board[i:i + IN_ROW, j]
                if (board_slice == 1).all():
                    res = 1
                elif (board_slice == -1).all():
                    res = -1
        # Check diagonal slices
        for i in range(BOARD_HEIGHT - IN_ROW + 1):
            for j in range(BOARD_WIDTH - IN_ROW + 1):
                board_slice = self.board[i:i + IN_ROW, j:j + IN_ROW].diagonal()
                if (board_slice == 1).all():
                    res = 1
                elif (board_slice == -1).all():
                    res = -1
        # Check transposed diagonal slices
        flipped = np.fliplr(self.board)
        for i in range(BOARD_HEIGHT - IN_ROW + 1):
            for j in range(BOARD_WIDTH - IN_ROW + 1):
                board_slice = flipped[i:i + IN_ROW, j:j + IN_ROW].diagonal()
                if (board_slice == 1).all():
                    res = 1
                elif (board_slice == -1).all():
                    res = -1
        if (self.board != 0).all():
            res = 0
        return res

    def get_state(self) -> np.ndarray:
        """Returns the array with the layered representation of the game board.
        The shape of the array is (Game.board_height, Game.board_width, Game.num_actions).

        Returns:
            np.ndarray: Representation of the board.
        """
        result = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, NUM_LAYERS))
        for (i, j), cell_value in np.ndenumerate(self.board):
            if cell_value == 1:
                result[i, j, 0] = 1
            elif cell_value == -1:
                result[i, j, 1] = 1
        result[:, :, 2] = self.get_current_move()
        result[:, :, 3] = np.array(self.board).reshape(
            (BOARD_HEIGHT, BOARD_WIDTH))
        return result

    def copy(self):
        """Copies the position.

        Returns:
            Position: Independent copy of the position.
        """
        return Position(self.board.copy())

    def visualize(self) -> str:
        """Returns human-readable representation of the board.

        Returns:
            str: Board image.
        """

        def symbol_from_int(cell_value: int) -> str:
            return {-1: "O", 0: ".", 1: "X"}[cell_value]

        board_string = np.vectorize(symbol_from_int)(self.board)
        return "\n".join(["".join(x) for x in board_string])


def position_from_image(pos: Image) -> Position:
    """Converts the image of a position into a Position object.

    Args:
        pos (Image): The image of a position.

    Returns:
        Position: Position object corresponding to the given image.
    """
    return Position(
        np.array([int(x) - 1 for x in pos]).reshape(BOARD_HEIGHT, BOARD_WIDTH))


START_POSITION: Position = Position(np.zeros((BOARD_HEIGHT, BOARD_WIDTH)))


class Game:
    """Class containing information about the game rules, the game in progress, etc.

    Raises:
        IndexError: Raises IndexError if the action to be commited is illegal.
    """

    num_actions: int = NUM_ACTIONS  # Number of actions possible in the game
    board_height = BOARD_HEIGHT
    board_width = BOARD_WIDTH
    num_layers = NUM_LAYERS

    def __init__(self, position=START_POSITION):
        self.position: Position = position.copy()  # Current in-game position

    def is_finished(self) -> bool:
        """Returns True if the game is over.

        Returns:
            bool: State of the game.
        """
        return self.position.get_winner() != 2

    def get_winner(self) -> int:
        """Returns the integer according to the result of the game.
        1 if the first pleyer won,
        0 if the game is tied,
        -1 if the second player won,
        2 if the game is not finished.

        Returns:
            int: The result of the game.
        """
        return self.position.get_winner()

    def get_actions(self) -> ndarray:
        """Returns the list of actions that are possible from the current position.

        Returns:
            ndarray: Array with the ids of all legal actions.
        """
        return np.array(
            np.where(
                self.position.board.reshape(Game.num_actions) == 0)).reshape(-1)

    def get_current_move(self) -> int:
        """Returns the integer representing the player that is to move next:
        1 corresponds to the player that moved first, -1 to the player that moved second.

        Returns:
            int
        """
        return self.position.get_current_move()

    def copy(self):
        """Returns the copy of the game.

        Returns:
            Game: The independend copy of the Game object.
        """
        new_game = Game()
        new_game.num_actions = self.num_actions
        new_game.position = self.position.copy()
        return new_game

    def commit_action(self, action: int):
        """Makes a move with id equal to `action`.

        Args:
            action (int): The id of the action to be commited.

        Raises:
            IndexError: Raises `IndexError` if the action to be commited is illegal.
        """
        if self.position.board.reshape(-1)[action] != 0:
            raise IndexError(f"The {action}-th cell is already taken")
        self.position.board.reshape(-1)[action] = self.get_current_move()


def augment_data(source_tensor: ndarray, y_act: ndarray,
                 y_val: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """Augments data generated through self-play by board rotations, flips, etc.

    Args:
        source_tensor (ndarray): Tensor with the positions information.
        y_act (ndarray): Tensor with the information about actions.
        y_val (ndarray): Tensor with the information about expected outcome.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: Augmented `(source_tensor, y_act, y_val)`
    """
    batch_size = source_tensor.shape[0]
    y_act = y_act.reshape((-1, 3, 3))
    result_x: ndarray = np.zeros((batch_size * 4, 3, 3, 4))
    result_y: ndarray = np.zeros((batch_size * 4, 3, 3))
    result_y_val: ndarray = np.zeros((batch_size * 4, 3))

    result_x[:batch_size] = source_tensor
    result_y[:batch_size] = y_act
    result_y_val[:batch_size] = y_val

    result_x[batch_size:batch_size * 2] = np.rot90(source_tensor,
                                                   k=1,
                                                   axes=(1, 2))
    result_y[batch_size:batch_size * 2] = np.rot90(y_act, k=1, axes=(1, 2))
    result_y_val[batch_size:batch_size * 2] = y_val

    result_x[batch_size * 2:batch_size * 3] = np.rot90(source_tensor,
                                                       k=2,
                                                       axes=(1, 2))
    result_y[batch_size * 2:batch_size * 3] = np.rot90(y_act, k=2, axes=(1, 2))
    result_y_val[batch_size * 2:batch_size * 3] = y_val

    result_x[batch_size * 3:] = np.rot90(source_tensor, k=3, axes=(1, 2))
    result_y[batch_size * 3:] = np.rot90(y_act, k=3, axes=(1, 2))
    result_y_val[batch_size * 3:] = y_val

    return result_x, result_y.reshape((-1, 9)), result_y_val


def test_position():
    """Runs tests on the `Position` class. Supposed to be run automatically.
    """
    pos = Position(np.array([
        [0, 0, 1],
        [0, 1, -1],
        [1, -1, 0],
    ]))
    assert pos.get_winner() == 1
    assert pos.get_current_move() == -1
    assert pos.to_image() == "112120201"
    assert (position_from_image(pos.to_image()).board == pos.board).all()
    pos = Position(np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]))
    assert pos.get_winner() == 2
    assert pos.get_current_move() == 1
    pos.get_state()
    pos2 = pos.copy()
    pos2.board[0, 0] = 1
    assert not (pos.board == pos2.board).all()
    pos = Position(np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]))
    assert pos.get_winner() == 1


def test_game():
    """Runs tests on the `Game` class. Supposed to be run automatically.
    """
    game = Game()
    assert not game.is_finished()
    assert game.get_winner() == 2
    print(game.get_actions().shape)
    print((Game.num_actions,))
    assert game.get_actions().shape == (Game.num_actions,)
    assert (game.get_actions() == np.arange(game.num_actions)).all()
    game.commit_action(0)
    assert not game.is_finished()
    assert (game.get_actions() == np.arange(1, game.num_actions)).all()
    game.commit_action(1)
    game.commit_action(4)
    game.commit_action(2)
    game.commit_action(8)
    assert game.get_winner() == 1
    assert game.is_finished()
