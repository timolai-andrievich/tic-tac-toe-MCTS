"""A small script that generates through self-play and prints it."""
import os
import sys
import glob

from tic_tac_toe_mcts.player import MctsPlayer
from tic_tac_toe_mcts.config import Config
from tic_tac_toe_mcts.policy import Model
from tic_tac_toe_mcts.game import Game


def print_board(game: Game):
    """Prints the game board to the stdandart output.

    Args:
        game (Game): The game to be printed.
    """
    print(game.position.visualize())
    print()


def main():
    """The entry point of the program
    """
    files = list(filter(os.path.isfile, glob.glob('./models/*.pt')))
    files.sort(key=os.path.getmtime)
    model_file = files[-1]
    config = Config()
    if not files:
        print('No model files found in ./models/', file=sys.stderr)
        sys.exit(1)
    model = Model(config, model_file)
    player = MctsPlayer(model, config, 0)
    game = Game()
    while not game.is_finished():
        action = player.get_action(game)
        game.commit_action(action)
        print_board(game)


if __name__ == '__main__':
    main()
