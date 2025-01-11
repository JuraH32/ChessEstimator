import argparse
import os
import pickle
from concurrent.futures.thread import ThreadPoolExecutor

import chess

from model.util import board_to_array


def process_file(file):
    # print(f"Processing {file}")
    with open(file, 'rb') as f:
        game_info = pickle.load(f)

    positions = []
    board = chess.Board()

    moves = game_info.get('Moves', [])

    for move in moves:
        board.push(chess.Move.from_uci(move))
        positions.append(board_to_array(board))

    game_info['Positions'] = positions

    with open(file, 'wb') as f:
        pickle.dump(game_info, f)

def main(data_dir: str):
    if not data_dir.endswith("/"):
        data_dir += "/"

    files = [f"{data_dir}{f}" for f in os.listdir(data_dir)]

    for i, file in enumerate(files):
        if i % 1000 == 0:
            print(f"Processed {i} files")
        process_file(file)

    # with ThreadPoolExecutor() as executor:
    #     executor.map(process_file, files)

    print("Finished converting all files")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--data_dir", type=str, default="data/unpacked_games")

    args = argparser.parse_args()

    main(args.data_dir)