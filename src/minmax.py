import numpy as np
from numpy import ndarray

class MinMax:
    def __init__(self):
        self.cache = {}
    
    def get_winner(self, board: ndarray):
        slices = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]
        for slice in slices:
            if (board.reshape(-1)[slice] == 1).all(): return 1
            elif (board.reshape(-1)[slice] == -1).all(): return -1
        if (board != 0).all(): return 0
        return 2

    def current_move(self, board: ndarray):
        return 1 if (board != 0).sum() % 2 == 0 else -1

    def legal_actions(self, board: ndarray):
        return np.array(np.where(board.reshape(-1) == 0)).reshape(-1)

    def pos_with_move(self, board: ndarray, move):
        nb = board.copy()
        nb[move] = self.current_move(board)
        return nb

    def evaluate(self, board: ndarray):
        key = ''.join(map(str, map(int, board)))
        if key not in self.cache:
            if self.get_winner(board) != 2:
                self.cache[key] = self.get_winner(board)
            else:
                la = self.legal_actions(board)
                evals = np.array([self.evaluate(self.pos_with_move(board, i)) for i in la])
                self.cache[key] = (evals * self.current_move(board)).max() * self.current_move(board)
        return self.cache[key]
    
    def policy_function(self, pos):
        board = pos.board.reshape(9)
        legal_actions = self.legal_actions(board)
        actions = np.zeros(9)
        values = np.zeros(3)
        evaluation = self.evaluate(board)
        values[evaluation] = 1
        for i in legal_actions:
            if self.evaluate(self.pos_with_move(board, i)) == evaluation:
                actions[i] = 1
        return actions / actions.sum(), values
