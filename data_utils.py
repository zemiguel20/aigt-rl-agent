import numpy as np


def read_objectives(filename):
    with open(filename) as file:
        lines = [line.strip() for line in file]

    i = 0
    shapes = []
    while i < len(lines):
        shape = []

        # shapes are separated by blank lines
        while i < len(lines) and lines[i].strip() != '':
            shape_line = []
            for char in lines[i].strip():
                shape_line.append(char == 'x')
            shape.append(shape_line)
            i += 1

        shapes.append(np.transpose(np.array(shape)))
        i += 1

    return shapes


def onehot_encode_board(board: np.ndarray, playerid: int):
    board_flatened = board.flatten()
    free_positions = (board_flatened == 0).astype(float)
    p1_positions = (board_flatened == playerid).astype(float)
    return np.concatenate((free_positions, p1_positions))


def onehot_enconde_move(move):
    return (move[0]/6, move[1]/6)  # normalize
