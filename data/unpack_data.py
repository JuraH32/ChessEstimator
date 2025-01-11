import argparse
import os
import time
import pickle
import re

import chess
import chess.pgn

from model.util import board_to_array


def game_to_pgn_string(game):
    # Aux function to convert a game object to a pgn string
    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    pgn_string = game.accept(exporter)
    return pgn_string


def parse_game(game, high_memory=False):
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
        if high_memory:
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
        if high_memory:
            positions = positions[:150]

        moves = moves[:150]

    if len(clocks) != len(moves):
        print(len(clocks), len(moves))
        print(game_to_pgn_string(game))
        return -1
    game_info = {
        "WhiteElo": game.headers.get("WhiteElo", None),
        "BlackElo": game.headers.get("BlackElo", None),
        "Result": game.headers.get("Result", None),
        "Clocks": clocks,
        "Time": time_control,
        "Moves": moves
    }

    if high_memory:
        game_info["Positions"] = positions

    return game_info


def process_pgn_file(filename, high_memory=False):
    """Processes each game in a PGN file."""
    with open(filename) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            game_info = parse_game(game, high_memory)
            if game_info:
                yield game_info


def main(file_path: str, files_dir: str, max_games: int, out_dir: str, high_memory: bool = False):
    file_path = file_path
    max_games = max_games
    out_dir = out_dir + "/" if not out_dir.endswith("/") else out_dir
    game_count = 0
    start = time.time()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # for game_info in process_pgn_stream(file_path):
    files = [file_path] if not files_dir else [f"{files_dir}{f}" for f in os.listdir(files_dir)]

    for file in files:
        for game_info in process_pgn_file(file, high_memory):
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
    argparser.add_argument("--out_dir", type=str, default="unpacked_games/")
    argparser.add_argument("--high_memory", action="store_true")

    args = argparser.parse_args()

    main(args.file_path, args.files_dir, args.max_games, args.out_dir, args.high_memory)
