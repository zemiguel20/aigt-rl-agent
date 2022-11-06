#! /usr/bin/env -S python -u
import cProfile
import os
import time

import numpy as np
from bandit_agent import BanditAgent

from data_utils import onehot_encode_board, onehot_enconde_move, read_objectives
from game import Game
import tensorflow as tf
import multiprocessing as mp
from mcts_agent import MctsAgent
from random_agent import RandomAgent

from rl_agent import RLAgent
from score import ScoreAgent


SIZE = 7
MCTS_ITERATION = 50
BANDIT_ITERATIONS = 250
OBJECTIVES = read_objectives("shapes-L.txt")
SELFPLAY_ITR = 1
DISCOUNT_FACTOR = 0.9


def main():

    # CREATE EMPTY NN
    nn: tf.keras.Sequential = init_nn()

    # GENERATE INITIAL DATA (MILESTONE 1)
    if not os.path.exists('supervised_model'):
        work = []
        for _ in range(1):
            players = [RandomAgent(1), RandomAgent(2)]
            work.append((SIZE, OBJECTIVES, players))
            players = [RandomAgent(2), RandomAgent(1)]
            work.append((SIZE, OBJECTIVES, players))
            players = [BanditAgent(BANDIT_ITERATIONS, 1),
                       BanditAgent(BANDIT_ITERATIONS, 2)]
            work.append((SIZE, OBJECTIVES, players))
            players = [BanditAgent(BANDIT_ITERATIONS, 2),
                       BanditAgent(BANDIT_ITERATIONS, 1)]
            work.append((SIZE, OBJECTIVES, players))
            players = [MctsAgent(MCTS_ITERATION, 1),
                       MctsAgent(MCTS_ITERATION, 2)]
            work.append((SIZE, OBJECTIVES, players))
            players = [MctsAgent(MCTS_ITERATION, 2),
                       MctsAgent(MCTS_ITERATION, 1)]
            work.append((SIZE, OBJECTIVES, players))
            players = [ScoreAgent(0, 1), ScoreAgent(0, 2)]
            work.append((SIZE, OBJECTIVES, players))
            players = [ScoreAgent(0, 2), ScoreAgent(0, 1)]
            work.append((SIZE, OBJECTIVES, players))

        pool = mp.Pool(12)
        results = pool.starmap(play_game, work)
        pool.close()

        learn(nn, results)
        nn.save('supervised_model')

    # TEST NEURAL NETWORK AGAINST AGENTS (MILESTONE 1)
    nn = tf.keras.models.load_model('supervised_model')

    players = [RLAgent(MCTS_ITERATION, 1, nn), RandomAgent(2)]
    play_game(SIZE, OBJECTIVES, players, 'final')
    players = [RLAgent(MCTS_ITERATION, 1, nn),
               BanditAgent(BANDIT_ITERATIONS, 2)]
    play_game(SIZE, OBJECTIVES, players, 'final')
    players = [RLAgent(MCTS_ITERATION, 1, nn), MctsAgent(MCTS_ITERATION, 2)]
    play_game(SIZE, OBJECTIVES, players, 'final')
    players = [RLAgent(MCTS_ITERATION, 1, nn), ScoreAgent(0, 2)]
    play_game(SIZE, OBJECTIVES, players, 'final')

    # REINFORCEMENT AGENT SELF PLAY LEARNING (MILESTONE 2)
    if not os.path.exists('learned_model'):
        start = time.perf_counter()
        for i in range(SELFPLAY_ITR):
            if i % 2 == 0:
                players = [RLAgent(MCTS_ITERATION, 1, nn),
                           RLAgent(MCTS_ITERATION, 2, nn)]
            else:
                players = [RLAgent(MCTS_ITERATION, 2, nn),
                           RLAgent(MCTS_ITERATION, 1, nn)]

            result = play_game(SIZE, OBJECTIVES, players)
            learn(nn, result[0], result[1])

        end = time.perf_counter()
        print(f'{(end - start):.2f} seconds')
        nn.save('learned_model')

    # TEST NEURAL NETWORK AGAINST AGENTS (MILESTONE 2)
    nn = tf.keras.models.load_model('learned_model')

    players = [RLAgent(MCTS_ITERATION, 1, nn), RandomAgent(2)]
    play_game(SIZE, OBJECTIVES, players, 'final')
    players = [RLAgent(MCTS_ITERATION, 1, nn),
               BanditAgent(BANDIT_ITERATIONS, 2)]
    play_game(SIZE, OBJECTIVES, players, 'final')
    players = [RLAgent(MCTS_ITERATION, 1, nn), MctsAgent(MCTS_ITERATION, 2)]
    play_game(SIZE, OBJECTIVES, players, 'final')
    players = [RLAgent(MCTS_ITERATION, 1, nn), ScoreAgent(0, 2)]
    play_game(SIZE, OBJECTIVES, players, 'final')


def play_game(boardsize, objectives, players, print_board=None):
    print(str(os.getpid())+': GAME STARTED')
    game = Game.new(boardsize, objectives, players, print_board == 'all')
    winner = game.play()

    if print_board == 'final':
        game.print_result(winner)

    winner_id = 0 if winner is None else winner.id
    return (winner_id, game.history)


def init_nn():
    # INPUT: 49 (board free positions) + 49 (board agent moves) + 2 (next action position)
    # OUTPUT: Q value of (state|action)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=100, dtype=tf.dtypes.float16),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model


def learn(nn: tf.keras.Sequential, results):

    print('Learning Started')

    train_input = []
    train_output = []

    i = 0
    for (winner_id, game_history) in results:
        i += 1
        print('Processing Result '+str(i))

        # P1 history in reverse order
        p1_history = list(filter(lambda entry: entry[1] == 1, game_history))
        p1_history = list(reversed(p1_history))
        p1_rewards = [0] * len(p1_history)

        # P2 history in reverse order
        p2_history = list(filter(lambda entry: entry[1] == 2, game_history))
        p2_history = list(reversed(p2_history))
        p2_rewards = [0] * len(p2_history)

        # Set positive reward for winner and negative reward for loser.
        # No reward if draw
        if winner_id == 1:
            p1_rewards[0] = 1
            p2_rewards[0] = -1
        elif winner_id == 2:
            p1_rewards[0] = -1
            p2_rewards[0] = 1

        # Backwards propagate rewards with discount factor
        for i in range(1, len(p1_rewards)):
            p1_rewards[i] += p1_rewards[i - 1] * DISCOUNT_FACTOR
        for i in range(1, len(p2_rewards)):
            p2_rewards[i] += p2_rewards[i - 1] * DISCOUNT_FACTOR

        # Encode P1 history and add to training data
        for i in range(len(p1_history)):
            input_state = onehot_encode_board(p1_history[i][0].board, 1)
            input_action = onehot_enconde_move(p1_history[i][2])
            input = np.concatenate((input_state, input_action))
            train_input.append(input)
            train_output.append(p1_rewards[i])
        # Encode P2 history and add to training data
        for i in range(len(p2_history)):
            input_state = onehot_encode_board(p2_history[i][0].board, 1)
            input_action = onehot_enconde_move(p2_history[i][2])
            input = np.concatenate((input_state, input_action))
            train_input.append(input)
            train_output.append(p2_rewards[i])

    train_input = np.array(train_input, dtype=np.float16)
    train_output = np.array(train_input, dtype=np.float16)

    nn.fit(train_input, train_output)


if __name__ == '__main__':
    main()
