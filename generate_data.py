from itertools import starmap
from multiprocessing import Pool
from time import perf_counter
from bandit_agent import BanditAgent
from data_utils import encode_game_data, read_objectives, data_to_string
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
        map = starmap(play_game, work)
        for game in map:
            game()
    else:
        # you probably shouldn't set parallel to a value larger than the
        # number of cores on your CPU, as otherwise agents running in parallel
        # may compete for the time available during their turn
        with Pool(parallel) as pool:
            map = pool.starmap(play_game, work)
        pool.close()

    end = perf_counter()
    print(f'{(end - start):.2f} seconds')


def play_game(boardsize, objectives, players=None):
    game = Game.new(boardsize, objectives, players, False)
    winner = game.play()

    data = encode_game_data(game.board, winner)
    data_str = data_to_string(data)

    file = open('data.txt', 'a')
    file.write(data_str)
    file.write('\n')
    file.close()



