import os
import random

import chess
import chess.engine
from chess import Board

from app.rating_model import parse_clock


class ChessAgent:
    def get_move(self, board, rating_estimate, time_limit=None):
        pass

class ChessAgentMaia:
    def __init__(self, weights_path: str):
        self.engine_path = "maia/lc0"
        self.weights_path = weights_path
        self.engine = None
        self.start_engine()

    def start_engine(self, from_postion=None):
        # Start the engine process with Maia weights
        arguments = [self.engine_path, f"--weights={self.weights_path}", "--backend=blas"]

        self.engine = chess.engine.SimpleEngine.popen_uci(
            arguments
        )

    def stop_engine(self):
        if self.engine:
            self.engine.quit()

    def get_move(self, board: chess.Board, time_limit=1):
        if not self.engine:
            raise Exception("Engine is not running. Call start_engine() first.")

        # Calculate best move for the current board state
        result = self.engine.play(board, chess.engine.Limit(time=time_limit, nodes=1))
        return result.move

    def __del__(self):
        self.stop_engine()

class ChessAgentMaiaMultiple(ChessAgent):
    def __init__(self, clock):
        self.agents = self.load_agents()
        self.clock, self.increment = parse_clock(clock)

    def load_agents(self) -> dict:
        # load maia agents
        engine_paths = "maia"

        agents = {}

        for path in os.listdir(engine_paths):
            if path.endswith(".pb.gz"):
                rating = int(path.split(".")[0])
                agents[rating] = ChessAgentMaia(f"{engine_paths}/{path}")

        return agents

    def get_best_agent(self, rating_estimate) -> ChessAgent.__class__:
        best_diff = float('inf')
        best_agent = None

        for rating in sorted(self.agents.keys(), reverse=True):
            diff = abs(rating_estimate - rating)
            if diff < best_diff:
                best_diff = diff
                best_agent = rating

        return best_agent

    def get_move(self, board, rating_estimate, time_limit=None):
        if time_limit is None:
            time_limit = self.increment + self.clock / 60

        best_engine_rating = self.get_best_agent(rating_estimate)
        print("Using engine with rating: ", best_engine_rating)
        engine = self.agents[best_engine_rating]
        return engine.get_move(board, time_limit)

class ChessAgentRandom(ChessAgent):
    def get_move(self, board: Board, rating_estimate, time_limit=None):
        # Random random move
        bot_move = random.choice(list(board.legal_moves))
        return bot_move.uci()
