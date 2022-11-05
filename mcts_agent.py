# Put your name and student ID here before submitting!

from mc_tree import Node
from random_agent import RandomAgent
import game as g
import copy
import time

LOSS_SCORE = 0
DRAW_SCORE = 0.5
WIN_SCORE = 1


class MctsAgent:
    def __init__(self, iterations, id):
        self.iterations = iterations
        self.id = id

    def make_move(self, game):
        start = time.perf_counter()

        self.root = Node(copy.deepcopy(game.board),
                         who_is_making_the_next_move=self.id)
        self.root.expand()

        i = 0
        while i < self.iterations:
            current = self.root

            while not current.is_leaf():
                current = current.get_best_child(self.root.visits)

            if current.visits > 0 and not current.is_final_state:
                current.expand()
                current = current.children[0]

            score = self.rollout(current, game.objectives)
            current.backpropagate(score)

            i += 1

        # print(self.root.one_level_of_tree_as_string())
        # print(self.root.tree_as_string())
        # print(len(self.root.children))
        # print(self.root.count_nodes())

        move = self.root.get_best_move()

        # print(move)

        return move

    def rollout(self, node, objectives):
        rollout_game = g.Game.from_board(copy.deepcopy(node.board), objectives,
                                         [RandomAgent(node.who_is_making_the_next_move),
                                          RandomAgent(3 - node.who_is_making_the_next_move)], None)

        score = LOSS_SCORE
        if rollout_game.victory(node.move, 3 - node.who_is_making_the_next_move):
            # handle active player won
            node.is_final_state = True
            if 3 - node.who_is_making_the_next_move == self.id:
                score = WIN_SCORE
        else:
            winner = rollout_game.play()
            if winner == None:
                score = DRAW_SCORE
            elif winner.id == self.id:
                score = WIN_SCORE

        return score

    def __str__(self):
        return f'Player {self.id} (MCTSAgent)'
