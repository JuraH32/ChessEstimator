import argparse
import os
import time
import pickle
import re
import torch

import chess
import chess.pgn


def board_to_array(board):
    """Converts a chess board into a 12-layer 8x8 tensor representing the piece positions."""
    # Board('rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 0 1')
    board_array = torch.zeros((12, 8, 8), dtype=torch.float32)
    piece_map = board.piece_map()
    # {63: Piece.from_symbol('r'), 62: Piece.from_symbol('n')
    for square, piece in piece_map.items():
        # 1 pawn, 2 knight, 3 bishop, 4 rook, 5 queen, 6 king
        # Piece color is true if white
        # White pieces 0-5, black 6-11
        index = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        row, col = divmod(square, 8)
        board_array[index, 7 - row, col] = 1
    return board_array


def game_to_pgn_string(game):
    # Aux function to convert a game object to a pgn string
    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    pgn_string = game.accept(exporter)
    return pgn_string


def parse_game(game):
    """Parses game object to extract evaluations, clocks and positions."""
    time_control = game.headers.get('TimeControl', '')

    board = game.board()
    moves = []
    # evaluations = []
    clocks = []
    positions = []
    node = game
    # Take in the mainline, annotated pgn might have multiple variations
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        board.push(move)
        positions.append(board_to_array(board))
        moves.append(move.uci())
        comment = next_node.comment

        clock_match = re.search(r"\[%clk\s+([^\]]+)\]", comment)
        if clock_match:
            clocks.append(clock_match.group(1))

        node = next_node

    if not clocks:
        return None  # Skip games without evaluations or clock times (correspondence chess)

    # Some games did have evals after 150
    if len(clocks) > 150:
        # Evals stop at move 150
        # evaluations = evaluations[:150]
        clocks = clocks[:150]
        positions = positions[:150]

    if len(clocks) != len(positions):
        print(len(clocks), len(positions))
        print(game_to_pgn_string(game))
        return -1
    return {
        "WhiteElo": game.headers.get("WhiteElo", None),
        "BlackElo": game.headers.get("BlackElo", None),
        "Result": game.headers.get("Result", None),
        "Clocks": clocks,
        "Positions": positions,
        "Time": time_control,
        "Moves": moves
    }


def process_pgn_file(filename):
    """Processes each game in a PGN file."""
    with open(filename) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            game_info = parse_game(game)
            if game_info:
                yield game_info


def main(file_path: str, files_dir: str, max_games: int, out_dir: str):
    file_path = file_path
    max_games = max_games
    out_dir = out_dir
    game_count = 0
    start = time.time()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # for game_info in process_pgn_stream(file_path):
    files = [file_path] if not files_dir else [f"{files_dir}{f}" for f in os.listdir(files_dir)]

    for file in files:
        for game_info in process_pgn_file(file):
            if game_count % 1000 == 0:
                print(game_count)
            if game_info == -1:
                print("error", game_count)
                continue
            filename = f"{out_dir}lichess_db_standard_rated_{game_count}.pkl"
            with open(filename, 'wb') as file:
                pickle.dump(game_info, file)
            game_count += 1
            if max_games != 0 and game_count >= max_games:
                print("Finished saving data")
                break
    end = time.time()
    print(file_path)
    print("seconds: ", round(end - start))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--file_path", type=str, default="games/2024-09_games_60+0.pgn")
    argparser.add_argument("--files_dir", type=str, default=None)
    argparser.add_argument("--max_games", type=int, default=0)
    argparser.add_argument("--out_dir", type=str, default="processed_games/")

    args = argparser.parse_args()

    main(args.file_path, args.files_dir, args.max_games, args.out_dir)
