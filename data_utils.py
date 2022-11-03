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


def save_weights_to_file(nn):
    # TODO: implement
    return


def load_weights_from_file(nn):
    # TODO: implement
    return


def encode_game_data(board, winner):
    board_flatened = board.board.flatten()
    free_positions = (board_flatened == 0).astype(int)
    p1_positions = (board_flatened == 1).astype(int)
    result = 0 if winner is None else winner.id
    encoded_data = (free_positions, p1_positions, result)
    return encoded_data


def data_to_string(data):
    free_positions = ' '.join(map(str, data[0]))
    p1_positions = ' '.join(map(str, data[1]))
    result = str(data[2])
    data_str = [free_positions, ' | ', p1_positions, " | ", result]
    data_str = ''.join(data_str)
    return data_str


def data_from_string(data_str):
    split = data_str.split('|')
    split = [string.strip() for string in split]  # remove lead and trail space
    free_positions = np.fromstring(split[0], dtype=int, sep=' ')
    p1_positions = np.fromstring(split[1], dtype=int, sep=' ')
    result = int(split[2])
    encoded_data = (free_positions, p1_positions, result)
    return encoded_data
