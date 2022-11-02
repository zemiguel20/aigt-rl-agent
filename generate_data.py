from itertools import starmap
from multiprocessing import Pool
from os import getpid
from time import perf_counter
from bandit_agent import BanditAgent
from data_save_load import read_objectives
from game import Game
from mcts_agent import MctsAgent
from random_agent import RandomAgent
from score import ScoreAgent


def generate_data():
    batch_count = 1
    iterations = 250
    size = 7
    objectives = read_objectives("shapes-L.txt")
    parallel = 12

    work = []

    for _ in range(batch_count):
        # MCTS vs Random
        players = [MctsAgent(iterations, 1), RandomAgent(2)]
        work.append((size, objectives, players))
        players = [RandomAgent(2), MctsAgent(iterations, 1)]
        work.append((size, objectives, players))
        # MCTS vs Bandit
        players = [MctsAgent(iterations, 1), BanditAgent(iterations, 2)]
        work.append((size, objectives, players))
        players = [BanditAgent(iterations, 2), MctsAgent(iterations, 1)]
        work.append((size, objectives, players))
        # MCTS vs Score
        players = [MctsAgent(iterations, 1), ScoreAgent(iterations, 2)]
        work.append((size, objectives, players))
        players = [ScoreAgent(iterations, 2), MctsAgent(iterations, 1)]
        work.append((size, objectives, players))
        # Score vs Random
        players = [ScoreAgent(iterations, 1), RandomAgent(2)]
        work.append((size, objectives, players))
        players = [RandomAgent(2), ScoreAgent(iterations, 1)]
        work.append((size, objectives, players))
        # Score vs Bandit
        players = [ScoreAgent(iterations, 1), BanditAgent(iterations, 2)]
        work.append((size, objectives, players))
        players = [BanditAgent(iterations, 2), ScoreAgent(iterations, 1)]
        work.append((size, objectives, players))
        # Bandit vs Random
        players = [BanditAgent(iterations, 1), RandomAgent(2)]
        work.append((size, objectives, players))
        players = [RandomAgent(2), BanditAgent(iterations, 1)]
        work.append((size, objectives, players))

    start = perf_counter()

    if parallel == None:
        starmap(play_game, work)
    else:
        # you probably shouldn't set parallel to a value larger than the
        # number of cores on your CPU, as otherwise agents running in parallel
        # may compete for the time available during their turn
        with Pool(parallel) as pool:
            pool.starmap(play_game, work)
        pool.close()

    end = perf_counter()
    print(f'{(end - start):.2f} seconds')


def play_game(boardsize, objectives, players=None):
    game = Game.new(boardsize, objectives, players, False)
    winner = game.play()

    # TODO: encode game data into a string

    file = open('data.txt', 'a')
    pid = getpid()
    file.write(str(pid) + "_____\n")
    file.close()
