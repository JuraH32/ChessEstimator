import random

import chess


class ChessEngine:
    def __init__(self):
        self.board = chess.Board()
        self.game_history = []

    def reset_game(self):
        self.board = chess.Board()
        self.game_history = []

    def get_board(self):
        return self.board

    def is_valid_move(self, move):
        return move in [move.uci() for move in self.board.legal_moves]

    def make_move(self, move):
        print("Making move", move)

        self.board.push(chess.Move.from_uci(move))
        self.game_history.append(move)

    def generate_random_move(self):
        # Random random move
        bot_move = random.choice(list(self.board.legal_moves))
        return bot_move.uci()

    def get_game_history(self):
        return self.game_history