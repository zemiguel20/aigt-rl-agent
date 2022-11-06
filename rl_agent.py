# Put your name and student ID here before submitting!

import time
import numpy as np
from mc_tree import Node
import copy
import tensorflow as tf
from data_utils import onehot_encode_board, onehot_enconde_move


class RLAgent:
    def __init__(self, iterations, id, nn):
        self.iterations = iterations
        self.id = id
        self.nn: tf.keras.Sequential = nn

    def make_move(self, game):

        start = time.perf_counter()

        self.root = Node(copy.deepcopy(game.board),
                         who_is_making_the_next_move=self.id)
        self.root.expand()

        for _ in range(self.iterations):
            current = self.root

            while not current.is_leaf():
                current = current.get_best_child(self.root.visits)

            if current.visits > 0 and not current.is_final_state:
                current.expand()
                current = current.children[0]

            # estimate Q value
            score = self.estimate(current)
            current.backpropagate(score)

        move = self.root.get_best_move()

        end = time.perf_counter()
        print(f'RLAgent make move benchmark: {(end - start):.2f} seconds')

        return move

    def estimate(self, state: Node):
        input_state = onehot_encode_board(state.board.board, self.id)
        input_action = onehot_enconde_move(state.move)
        input = np.concatenate((input_state, input_action), dtype=np.float16) 
        input_array = np.array([input]) # NN takes as input a list of inputs
        prediction = self.nn.predict(input_array, verbose=0)
        return prediction[0]

    def __str__(self):
        return f'Player {self.id} (RLAgent)'
