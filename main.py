#! /usr/bin/env -S python -u
from game import Game
from random_agent import RandomAgent
from score import ScoreAgent
from mcts_agent import MctsAgent
from bandit_agent import BanditAgent

import argparse
import time
import numpy as np
import multiprocessing as mp
from collections import Counter
from itertools import starmap


def main():

    games = 100
    iterations = 250
    size = 7
    objectives = read_objectives("shapes-L.txt")
    print_board = None
    parallel = 12

    work = []
    for i in range(games):
        # swap order every game
        if i % 2 == 0:
            players = [ScoreAgent(iterations, 1), MctsAgent(iterations, 2)]
        else:
            players = [MctsAgent(iterations, 2), ScoreAgent(iterations, 1)]

        work.append((size, objectives, players, print_board))

    start = time.perf_counter()

    # the tests can be run in parallel, or sequentially
    # it is recommended to only use the parallel version for large-scale testing
    # of your agent, as it is harder to debug your program when enabled
    if parallel == None:
        results = starmap(play_game, work)
    else:
        # you probably shouldn't set parallel to a value larger than the
        # number of cores on your CPU, as otherwise agents running in parallel
        # may compete for the time available during their turn
        with mp.Pool(parallel) as pool:
            results = pool.starmap(play_game, work)

    stats = Counter(results)
    end = time.perf_counter()

    print(f'{stats[1]}/{stats[2]}/{stats[0]} ({(end - start):.2f} seconds)')


def play_game(boardsize, objectives, players, print_board=None):
    game = Game.new(boardsize, objectives, players, print_board == 'all')
    winner = game.play()

    if print_board == 'final':
        game.print_result(winner)

    return 0 if winner == None else winner.id


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


if __name__ == '__main__':
    main()
