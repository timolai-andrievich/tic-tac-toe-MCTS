from collections import namedtuple
from typing import List, Tuple, Dict


TTT_NUM_ACTIONS: int = 9 # Game-specific 

class Position:
    '''Represents position in the game'''


START_POSITION: Position = None


class Game:
    '''Describes the logic of the game and provides appropriate interface'''
    def __init__(self):
        self._num_actions: int = TTT_NUM_ACTIONS # Number of actions possible in the game
        self._data: List[Tuple[Position, Dict[int, float]]] = [] # Stores data about analyzed positions: 
                                                                 # representation of a position that can be passed to a NN 
                                                                 # and a map from action number to a probability of this action
        self._position: Position = START_POSITION # Current in-game position
        self._positions: List[Position] = [] # List of all the positions reached
        
    def get_actions(self) -> List[int]:
        '''Returns the list of actions that are possible from the current position'''

    def get_data(self) -> List[Tuple[Position, Dict[int, float]]]:
        '''Returns data collected during self-play'''

    def store_data(self, data: Dict[int, float]):
        '''Adds data about the root of MCTS tree to the storage'''

    def is_terminal(self) -> bool:
        '''Returns true if the position of the game is terminal'''

    def get_winner(self) -> int:
        '''Returns 1 if player who moved first won, 0 if the game is a tie, -1 if the player who moved second won'''

    def get_current_move(self) -> int:
        '''Returns 1 if player who moved first should move now, -1 if the player who moved second should move now'''

    def copy(self) -> Game:
        '''Returns the copy of the game'''

    def commit_action(action: int):
        '''Makes a move according to the id of the action'''
    