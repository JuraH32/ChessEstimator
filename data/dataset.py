import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def time_to_seconds(time_str):
    """Converts a time string of the format 'HH:MM:SS' to seconds."""
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


class ChessGamesDataset(Dataset):
    def __init__(self, filenames, max_moves=100, ratings_mean=1650, ratings_std=433, clocks_mean=60, clocks_std=0.1):
        self.filenames = filenames
        self.max_moves = max_moves
        self.ratings_mean = ratings_mean
        self.ratings_std = ratings_std
        self.clocks_mean = clocks_mean
        self.clocks_std = clocks_std

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with open(self.filenames[idx], 'rb') as f:
            game_info = pickle.load(f)
        clocks = [time_to_seconds(c) for c in game_info.get('Clocks', [])]
        clocks = [(c - self.clocks_mean) / self.clocks_std for c in clocks]
        clocks = torch.tensor(clocks, dtype=torch.float)[:self.max_moves]
        # Ablation to set clocks to 0
        # clocks = torch.zeros_like(clocks)
        white = False
        if "white" in game_info:
            white = game_info["white"]
        last_rating = None
        if "rating_after_last_game" in game_info:
            last_rating = game_info["rating_after_last_game"]
            last_rating = (last_rating - self.ratings_mean) / self.ratings_std
            last_rating = torch.tensor(last_rating, dtype=torch.float)
        positions = torch.stack(game_info['Positions'])[:self.max_moves]
        white_elo, black_elo = float(game_info['WhiteElo']), float(game_info['BlackElo'])
        targets = torch.tensor([white_elo, black_elo], dtype=torch.float)
        targets = (targets - self.ratings_mean) / self.ratings_std

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
