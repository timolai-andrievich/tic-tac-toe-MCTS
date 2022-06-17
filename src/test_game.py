from Game import Game, Position, position_from_image

def test_position():
    pos = Position([1, -1, -1, 1, 0, 0, 1, 0, 0])
    assert(pos.get_winner() == 1)
    assert(pos.get_current_move() == -1)
    assert(pos.to_image() == '200211211')
    assert(position_from_image(pos.to_image()).board == pos.board)
    pos = Position([-1, -1, -1, 1, 0, 0, 1, 1, 0])
    assert(pos.get_winner() == -1)
    assert(pos.get_current_move() == 1)

def test_game():
    g = Game()
    g.commit_action(0)
    g.commit_action(1)
    g.commit_action(3)
    g.commit_action(2)
    g.commit_action(6)
    g.assign_scores()
    assert(g.get_current_move() == -1)
    assert(g.get_actions() == [4, 5, 7, 8])
    assert(g.is_terminal() == True)
    assert(g.get_scores() == [1, 1, 1, 1, 1])