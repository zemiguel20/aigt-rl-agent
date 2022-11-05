import copy
import math

UCB_PARAMETER = 0.75  # math.sqrt(2)


class Node:
    def __init__(self, board, parent=None, move=None, who_is_making_the_next_move=None):
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
            self.children.append(
                Node(new_board, self, move, 3 - self.who_is_making_the_next_move))

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
            score = child.visits  # 0 if child.visits == 0 else child.total_score / child.visits
            # if score == best_score:
            #     best_moves.append(child.move)
            if score > best_score:
                best_score = score
                best_move = child.move

        return best_move  # best_moves[random.randrange(0, len(best_moves))]

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

    def tree_as_string(self, tabs="", level=0):
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
