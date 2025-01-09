import pickle

import chess
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from util import board_to_array


def time_to_seconds(time_str):
    """Converts a time string of the format 'HH:MM:SS' to seconds."""
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


class ChessGamesDataset(Dataset):
    def __init__(self, filenames, max_moves=100, min_rating=900, max_rating=2400, clock=60):
        self.filenames = filenames
        self.max_moves = max_moves
        self.clock = clock
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.rating_range = max_rating - min_rating

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with open(self.filenames[idx], 'rb') as f:
            game_info = pickle.load(f)
        clocks = [time_to_seconds(c) for c in game_info.get('Clocks', [])]
        clocks = [c / self.clock for c in clocks]
        clocks = torch.tensor(clocks, dtype=torch.float)[:self.max_moves]
        # Ablation to set clocks to 0
        # clocks = torch.zeros_like(clocks)
        white = False
        if "white" in game_info:
            white = game_info["white"]
        last_rating = None
        if "rating_after_last_game" in game_info:
            last_rating = game_info["rating_after_last_game"]
            last_rating = (last_rating - self.min_rating) / self.rating_range
            last_rating = torch.tensor(last_rating, dtype=torch.float)

        if "Positions" not in game_info:
            moves = game_info.get('Moves', [])
            board = chess.Board()
            game_positions = []
            for move in moves:
                board.push(chess.Move.from_uci(move))
                game_positions.append(board_to_array(board))
        else:
            game_positions = game_info['Positions']

        positions = torch.stack(game_positions)[:self.max_moves]
        white_elo, black_elo = float(game_info['WhiteElo']), float(game_info['BlackElo'])
        targets = torch.tensor([white_elo], dtype=torch.float)
        targets = (targets - self.min_rating) / self.rating_range

        length = len(positions)
        initial_time, increment = map(int, game_info['Time'].split('+'))
        estimated_duration = initial_time + 40 * increment
        time_control = self.categorize_time_control(estimated_duration)

        result = None
        if "Result" in game_info:
            result = game_info["Result"]

        return {'positions': positions, 'clocks': clocks, 'targets': targets, 'length': length,
                'time_control': time_control,
                'white': white, 'last_rating': last_rating, 'result': result}

    def categorize_time_control(self, estimated_duration):
        # Categories based on time control in seconds on Lichess
        if estimated_duration < 29:
            return 'ultrabullet'
        elif estimated_duration < 179:
            return 'bullet'
        elif estimated_duration < 479:
            return 'blitz'
        elif estimated_duration < 1499:
            return 'rapid'
        else:
            return 'classical'

def collate_fn(batch):
    # prepare batches
    positions = pad_sequence([item['positions'] for item in batch], batch_first=True)
    clocks = pad_sequence([item['clocks'] for item in batch], batch_first=True)
    targets = torch.stack([item['targets'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.int)
    time_controls = [item['time_control'] for item in batch]
    white = torch.tensor([item['white'] for item in batch])
    last_rating = None
    if batch[0]['last_rating']:
        last_rating = torch.stack([item['last_rating'] for item in batch])
    if batch[0]['result']:
        results = [item['result'] for item in batch]
        return {'positions': positions, 'clocks': clocks, 'targets': targets, 'lengths': lengths, 'time_controls': time_controls, 'white': white, 'last_rating': last_rating, 'results': results}
    return {'positions': positions, 'clocks': clocks, 'targets': targets, 'lengths': lengths, 'time_controls': time_controls, 'white': white, 'last_rating': last_rating}
