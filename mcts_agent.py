# Put your name and student ID here before submitting!

from random_agent import RandomAgent
import game as g

import random, copy, time, math

UCB_PARAMETER = 0.75 #math.sqrt(2)
LOSS_SCORE = 0
DRAW_SCORE = 0.5
WIN_SCORE = 1

def get_other_id(active_id):
    if active_id == 1:
        return 2
    
    return 1

class Node:
    def __init__(self, board, parent = None, move = None, who_is_making_the_next_move = None):
        self.board = board
        self.parent = parent
        self.move = move
        self.total_score = 0
        self.visits = 0
        self.children = []
        self.who_is_making_the_next_move = who_is_making_the_next_move

        self.is_final_state = False
        if self.board.full():
            self.is_final_state = True

    def expand(self):
        for move in self.board.free_positions():
            new_board = copy.deepcopy(self.board)
            new_board.place(move, self.who_is_making_the_next_move)
            self.children.append(Node(new_board, self, move, get_other_id(self.who_is_making_the_next_move)))

    def get_ucb_score(self, root_visits):
        if self.visits == 0:
            return math.inf
        
        average_score = self.total_score / self.visits
        
        return average_score + UCB_PARAMETER * math.sqrt(math.log2(root_visits) / self.visits)

    def is_leaf(self):
        return not self.children or self.is_final_state
    
    def get_best_child(self, root_visits):
        best_child = None
        best_score = -math.inf

        for child in self.children:
            score = child.get_ucb_score(root_visits)

            if score == math.inf:
                return child
            
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def get_best_move(self):
        best_score = -math.inf
        best_move = None

        for child in self.children:
            score = child.visits #0 if child.visits == 0 else child.total_score / child.visits
            # if score == best_score:
            #     best_moves.append(child.move)
            if score > best_score:
                best_score = score
                best_move = child.move

        return best_move #best_moves[random.randrange(0, len(best_moves))]

    def backpropagate(self, score):
        self.total_score += score
        self.visits += 1

        if self.parent:
            self.parent.backpropagate(score)

    def one_level_of_tree_as_string(self):
        result = f" {self}"

        for child in self.children:
            result += f"  |{child}"
        
        return result

    def tree_as_string(self, tabs = "", level = 0):
        result = f"{tabs} {level} | {self}"

        for child in self.children:
            result += child.tree_as_string(tabs + "  ", level + 1)
        
        return result

    def count_nodes(self):
        count = 1

        for child in self.children:
            count += child.count_nodes()
        
        return count
    
    def __str__(self):
        return f'Node score: {self.total_score}, total: {self.visits}, move: {self.move}, is_final: {self.is_final_state} \n'


        
class MctsAgent:
    def __init__(self, iterations, id):
        self.iterations = iterations
        self.id = id

    def make_move(self, game):
        start = time.perf_counter()

        self.root = Node(copy.deepcopy(game.board), who_is_making_the_next_move = self.id)
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
        rollout_game = g.Game.from_board(copy.deepcopy(node.board), objectives, \
                        [RandomAgent(node.who_is_making_the_next_move), \
                         RandomAgent(get_other_id(node.who_is_making_the_next_move))], None)
                    
        score = LOSS_SCORE
        if rollout_game.victory(node.move, get_other_id(node.who_is_making_the_next_move)):
            # handle active player won
            node.is_final_state = True
            if get_other_id(node.who_is_making_the_next_move) == self.id:
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
