from hashlib import new
from typing import List, Tuple, Dict


NUM_ACTIONS: int = 9  # Game-specific
Image = str


class Position:
    """Represents position in the game"""

    def __init__(self, board: List[int]):
        self.board: List[int] = board

    def to_image(self) -> Image:
        """Returns the representation of the position that preserves all the information about the board"""
        res = ""
        for i in self.board:
            res += str(i + 1)
        return res

    def get_current_move(self) -> int:
        """Returns 1 if player who moved first should move now, -1 if the player who moved second should move now"""
        return 1 if ((9 - self.board.count(0)) % 2 == 0) else -1

    def get_winner(self) -> int:
        """Returns 1 if player who moved first won, 0 if the game is a tie, -1 if the player who moved second won"""
        slices = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 4, 8],
            [2, 4, 6],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
        ]
        res = 0
        for slice in slices:
            a, b, c = slice
            if self.board[a] == self.board[b] == self.board[c] != 0:
                res = self.board[a]
                return res

    def vectorize(self) -> List[float]:
        """Returns normalized vector that represents the position for NN training"""
        result = []
        for i in self.board:
            if i == 1:
                result += [1, 0, 0]
            elif i == 0:
                result += [0, 1, 0]
            elif i == -1:
                result += [0, 0, 1]
        return result

    def copy(self):
        return Position(self.board.copy())


def position_from_image(pos: Image) -> Position:
    """Returns the position from the image of the position"""
    return Position([int(x) - 1 for x in pos])


START_POSITION: Position = Position([0] * 9)


class Game:
    """Describes the logic of the game and provides appropriate interface"""

    def __init__(self):
        self._num_actions: int = NUM_ACTIONS  # Number of actions possible in the game
        self._position: Position = START_POSITION  # Current in-game position
        self._positions: List[Position] = []  # List of all the positions reached
        self._scores: Dict[Image, float]  # Evaluation scores of the positions

    def get_actions(self) -> List[int]:
        """Returns the list of actions that are possible from the current position"""
        result = []
        for i, v in enumerate(self._position.board):
            if not v:
                result += [i]
        return result

    def is_terminal(self) -> bool:
        """Returns true if the position of the game is terminal"""
        return self._position.get_winner() or self._position.board.count(0) == 0

    def get_winner(self) -> int:
        """Returns 1 if player who moved first won, 0 if the game is a tie, -1 if the player who moved second won"""
        if self._position.board.count(0) == 0:
            return 0
        return self._position.get_winner()

    def get_current_move(self) -> int:
        """Returns 1 if player who moved first should move now, -1 if the player who moved second should move now"""
        return self._position.get_current_move()

    def copy(self):
        """Returns the copy of the game"""
        new_game = Game()
        new_game._scores = self._scores
        new_game._num_actions = self._num_actions
        new_game._position = self._position
        new_game._positions = []
        for pos in self._position:
            new_game._positions.append(pos.copy())
        return new_game

    def commit_action(self, action: int):
        """Makes a move according to the id of the action"""
        if self._position.board[action] != 0:
            raise IndexError(f"The {action}-th cell is already taken")
        self._position.board[action] = self.get_current_move()
        self._positions.append(self._position.copy())

    def assign_scores(self):
        """Assigns the evaluation scores to the positions based on the winner of the game"""
        if not self.is_terminal():
            raise ValueError("The game is not finished yet")
        self._scores = [self.get_winner()] * self._positions.__len__()

    def get_scores(self) -> List[float]:
        """Returns the final evaluation scores for the positions in the game"""
        return self._scores