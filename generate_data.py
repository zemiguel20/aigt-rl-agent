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
    batch_count = 500
    bandit_itr = 500
    mcts_itr = 250
    size = 7
    objectives = read_objectives("shapes-L.txt")
    parallel = 12

    work = []

    for _ in range(batch_count):
        # MCTS vs Random
        players = [MctsAgent(mcts_itr, 1), RandomAgent(2)]
        work.append((size, objectives, players))
        players = [RandomAgent(2), MctsAgent(mcts_itr, 1)]
        work.append((size, objectives, players))
        # MCTS vs Bandit
        players = [MctsAgent(mcts_itr, 1), BanditAgent(bandit_itr, 2)]
        work.append((size, objectives, players))
        players = [BanditAgent(bandit_itr, 2), MctsAgent(mcts_itr, 1)]
        work.append((size, objectives, players))
        # MCTS vs Score
        players = [MctsAgent(mcts_itr, 1), ScoreAgent(0, 2)]
        work.append((size, objectives, players))
        players = [ScoreAgent(0, 2), MctsAgent(mcts_itr, 1)]
        work.append((size, objectives, players))
        # Score vs Random
        players = [ScoreAgent(0, 1), RandomAgent(2)]
        work.append((size, objectives, players))
        players = [RandomAgent(2), ScoreAgent(0, 1)]
        work.append((size, objectives, players))
        # Score vs Bandit
        players = [ScoreAgent(0, 1), BanditAgent(bandit_itr, 2)]
        work.append((size, objectives, players))
        players = [BanditAgent(bandit_itr, 2), ScoreAgent(0, 1)]
        work.append((size, objectives, players))
        # Bandit vs Random
        players = [BanditAgent(bandit_itr, 1), RandomAgent(2)]
        work.append((size, objectives, players))
        players = [RandomAgent(2), BanditAgent(bandit_itr, 1)]
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



