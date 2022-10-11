# -*- coding: utf-8 -*-
"""
Created on Mon Oct 04 20:16:15 2022

@author: mdazizulh
"""


# 3D Tic-Tac-Toe using Python

# import libraries
import numpy as np


class TicTacToe3D(object):
    
    #Check for possible winning combinarions
    possible_winning_combos = (
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14],
        [15, 16, 17], [18, 19, 20], [21, 22, 23], [24, 25, 26],

        [0, 3, 6], [1, 4, 7], [2, 5, 8], [9, 12, 15], [10, 13, 16],
        [11, 14, 17], [18, 21, 24], [19, 22, 25], [20, 23, 26],

        [0, 4, 8], [2, 4, 6], [9, 13, 17], [11, 13, 15], [18, 22, 26],
        [20, 22, 24],

        [0, 9, 18], [1, 10, 19], [2, 11, 20], [3, 12, 21], [4, 13, 22],
        [5, 14, 23], [6, 15, 24], [7, 16, 25], [8, 17, 26],

        [0, 12, 24], [1, 13, 25], [2, 14, 26], [6, 12, 18], [7, 13, 19],
        [8, 14, 20], [0, 10, 20], [3, 13, 23], [6, 16, 26], [2, 10, 18],
        [5, 13, 21], [8, 16, 24], [0, 13, 26], [2, 13, 24], [6, 13, 20],
        [8, 13, 18]
    )
    
    #Find a given key in a 3D array.
    def find(self, arr, key):
        cnt = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if cnt == key:
                        return (i, j, k)
                    cnt += 1
                    
    #Given a combination find the coordinates of each part.
    def find_combo(self, combo):
        r, c, h = combo
        r = self.find(self.np_board, r)
        c = self.find(self.np_board, c)
        h = self.find(self.np_board, h)
        return r, c, h
    
    #Retrieve moves for a player that are in winning combinations.
    def get_moves_by_combination(self, player):
        moves = []
        for combo in self.possible_winning_combos:
            move = []
            for cell in combo:
                r, c, h = self.find(self.np_board, cell)
                if self.np_board[r][c][h] == player:
                    move += [cell]
            moves += [move]
        return moves

    #Get the previous moves for the player    
    def get_moves(self, player):
        moves = []
        counter = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if self.np_board[i][j][k] == player:
                        moves += [counter]
                    counter += 1
        return moves

    #Find the list of available moves
    def possible_moves(self, player):
        
        return list(self.allowed_moves) + self.get_moves(player)

    
    #Check whether the game has finished or not
    @property
    def complete(self):
        for player in self.players:
            for combo in self.possible_winning_combos:
                combo_available = True
                for pos in combo:
                    if pos not in self.possible_moves(player):
                        combo_available = False
                if combo_available:
                    return self.winner is not None
        return True

    #Make a list of the winning positions
    @property
    def winning_combo(self):
        if self.winner:

            positions = self.get_moves(self.winner)
            for combo in self.possible_winning_combos:
                winner = combo
                for pos in combo:
                    if pos not in positions:
                        winner = None
                if winner:
                    return winner
        return None

    #Return the player who has won if the game has finished
    @property
    def winner(self):
        for player in self.players:
            positions = self.get_moves(player)
            for combo in self.possible_winning_combos:
                won = True
                for pos in combo:
                    if pos not in positions:
                        won = False
                if won:
                    return player

        return None

    
    #Check the player 1 is the winner or not
    def p1_won(self):
        return self.winner == self.player_1

    
    #Check the player 2 is the winner or not
    def p2_won(self):
        return self.winner == self.player_2

    
    #Check whether the game is tie or not
    def tied(self):
        return self.complete and self.winner is None

    
    def simple_heuristic(self):
        return self.check_available(self.player_1) - self.check_available(self.player_2)

    def find_value(self, key):
        b, r, c = self.find(self.np_board, key)
        return self.np_board[b][r][c]

    #Check for available wins
    def check_available(self, player):
        enemy = self.get_enemy(player)
        wins = 0
        for combo in self.possible_winning_combos:
            if all([self.find_value(x) == player or \
                    self.find_value(x) != enemy for x in combo]):
                wins += 1
        return wins

    def computers_move(self):
        """Initiates the process of attempting to find the best (or decent)
        move possible from the available positions on the board."""

        best_score = -1000
        best_move = None
        h = None
        win = False

        for move in self.allowed_moves:
            self.move(move, self.ai)
            if self.complete:
                win = True
                break
            else:
                h = self.minimax_alphaBeta(self.human, -1000, 1000)
                self.depth_count = 0
                if h >= best_score:
                    best_score = h
                    best_move = move
                    self.undo_move(move)
                else:
                    self.undo_move(move)

                # see if it blocks the player
                self.move(move, self.human)
                if self.complete and self.winner == self.human:
                    if 1001 >= best_score:
                        best_score = 1001
                        best_move = move
                self.undo_move(move)

        if not win:
            self.move(best_move, self.ai)
        self.human_turn = True

    #Recursive Minimax algorithm & Alpha-Beta prunning    
    def minimax_alphaBeta(self, player, a, b):
        
        if self.depth_count == self.difficulty:
            return self.simple_heuristic
        if self.depth_count <= self.difficulty:
            self.depth_count += 1
            if player == self.player_1:
                h = -1000
                for move in self.allowed_moves:
                    self.move(move, player)
                    if self.complete:
                        self.undo_move(move)
                        return 1000
                    else:
                        h = self.minimax_alphaBeta(self.player_1, a, b)
                        if h > a:
                            a = h
                            self.undo_move(move)
                        else:
                            self.undo_move(move)
                    if a >= b:
                        break
                return a
            else:
                h = 1000
                for move in self.allowed_moves:
                    self.move(move, player)
                    if self.complete:
                        self.undo_move(move)
                        return -1000
                    else:
                        h = self.minimax_alphaBeta(self.player_2, a, b)
                        if h < b:
                            b = h
                            self.undo_move(move)
                        else:
                            self.undo_move(move)
                    if a >= b:
                        break
                return b
        else:
            return self.simple_heuristic

    
    #Reverses a move
    def undo_move(self, position):
        self.allowed_moves += [position]
        self.allowed_moves.sort()
        i, j, k = self.find(self.np_board, position)
        self.np_board[i][j][k] = position

    def move(self, position, player):
        """Initiates a move on the given position.

        Args:
            position (int): Position on board to replace.
            player (str): Player to set piece to.
        """
        self.allowed_moves.remove(position)
        self.allowed_moves.sort()
        i, x, y = self.find(self.np_board, position)
        self.np_board[i][x][y] = player

    def get_enemy(self, player):
        """Returns the enemy of the player provided.

        Args:
            player (str): Player to get enemy of.
        """
        if player == self.player_1:
            return self.player_1
        else:
            return self.player_2

    def pl1(self):
        """Initiates the process of attempting to find the best (or decent)
        move possible from the available positions on the board."""

        best_score = -1000
        best_move = None
        h = None
        win = False

        for move in self.allowed_moves:
            self.move(move, self.player_1)
            if self.complete:
                win = True
                break
            else:
                h = self.minimax_alphaBeta(self.player_2, -1000, 1000)
                self.depth_count = 0
                if h >= best_score:
                    best_score = h
                    best_move = move
                    self.undo_move(move)
                else:
                    self.undo_move(move)

                # see if it blocks the player
                self.move(move, self.player_2)
                if self.complete and self.winner == self.player_2:
                    if 1001 >= best_score:
                        best_score = 1001
                        best_move = move
                self.undo_move(move)

        if not win:
            self.move(best_move, self.player_1)

        self.player_1_turn = False
        # if self.pl2_check:
        #     self.player_1_turn = True
        # else:
        #     self.player_1_turn = False

    def pl2(self):
        """Initiates the process of attempting to find the best (or decent)
        move possible from the available positions on the board."""

        best_score = -1000
        best_move = None
        h = None
        win = False

        for move in self.allowed_moves:
            self.move(move, self.player_2)
            if self.complete:
                win = True
                break
            else:
                h = self.minimax_alphaBeta(self.player_1, -1000, 1000)
                self.depth_count = 0
                if h >= best_score:
                    best_score = h
                    best_move = move
                    self.undo_move(move)
                else:
                    self.undo_move(move)

                # see if it blocks the player
                self.move(move, self.player_1)
                if self.complete and self.winner == self.player_1:
                    if 1001 >= best_score:
                        best_score = 1001
                        best_move = move
                self.undo_move(move)

        if not win:
            self.move(best_move, self.player_2)

        self.player_1_turn = True

    """3D TTT logic and underlying game state object.

    Attributes:
        board (np.ndarray)3D array for board state.
        difficulty (int): Ply; number of moves to look ahead.
        depth_count (int): Used in conjunction with ply to control depth.

    Args:
        player (str): Player that makes the first move.
        player_1 (Optional[str]): player_1's character.
        player_2 (Optional[str]): player_2's character.
        ply (Optional[int]): Number of moves to look ahead.
    """

    def __init__(self, board=None, player=-1, player_1=-1, player_2=1, ply=3):

        if board is not None:
            assert type(board) == np.ndarray, "Board must be a numpy array"
            assert board.shape == (3, 3, 3), "Board must be 3x3x3"
            self.np_board = board
        else:
            self.np_board = self.create_board()
        self.map_seq_to_ind, self.map_ind_to_seq = self.create_map()
        self.allowed_moves = list(range(3*3*3))
        self.difficulty = ply
        self.depth_count = 0
        self.pl2_check = False
        if player == player_1:
            self.player_1_turn = True
        else:
            self.player_1_turn = False
            self.pl2_check = True

        self.player_1 = player_1
        self.player_2 = player_2
        self.players = (self.player_1, self.player_2)

    def create_map(self):
        """Create a mapping between index of 3D array and list of sequence, and vice-versa.

        Args: None

        Returns:
            map_seq_to_ind (dict): Mapping between sequence and index.
            map_ind_to_seq (dict): Mapping between index and sequence.
        """
        a = np.hstack((np.zeros(9), np.ones(9), np.ones(9) * 2))
        a = a.astype(int)
        b = np.hstack((np.zeros(3), np.ones(3), np.ones(3) * 2))
        b = np.hstack((b, b, b))
        b = b.astype(int)
        c = np.array([0, 1, 2], dtype=int)
        c = np.tile(c, 9)
        mat = np.transpose(np.vstack((a, b, c)))
        ind = np.linspace(0, 26, 27).astype(int)
        map_seq_to_ind = {}
        map_ind_to_seq = {}
        for i in ind:
            map_seq_to_ind[i] = (mat[i][0], mat[i][1], mat[i][2])
            map_ind_to_seq[(mat[i][0], mat[i][1], mat[i][2])] = i

        return map_seq_to_ind, map_ind_to_seq

    def reset(self):
        """Reset the game state."""
        self.allowed_moves = list(range(pow(3, 3)))
        self.np_board = self.create_board()
        self.depth_count = 0

    @staticmethod
    def create_board():
        # Creates 3D numpy board
        np_board = np.zeros((3, 3, 3), dtype=int)
        return np_board

    
    #Primary Loop
    def play_game(self):
        while not self.complete:
                if self.player_1_turn:
                    self.pl1()
                else:
                    self.pl2()

        self.np_board[self.np_board > 1] = 0

        return self.np_board, self.winner
        


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '--player', dest='player', help='Player that plays first, 1 or -1', \
        type=int, default=-1, choices=[1, -1]
    )
    parser.add_argument(
        '--ply', dest='ply', help='Number of moves to look ahead', \
        type=int, default=6
    )
    args = parser.parse_args()

    brd, winner = TicTacToe3D(player=args.player, ply=args.ply).play_game()
    print("Board State: \n{}".format(brd))
    print("Winner of the Game: {}".format(winner))