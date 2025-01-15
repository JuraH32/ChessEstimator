import random

import torch
import numpy as np

from data.util import board_to_array
from model.predictor import ChessEloPredictor
from util import get_device

increment_rating_ranges = {
    "60+0": (900, 2400),
    "180+0": (1000, 2100),
    "300+0": (1000, 2100),
    "180+2": (1000, 2100),
}

def parse_clock(timeIncrement):
    if "+" in timeIncrement:
        clock, increment = timeIncrement.split("+")
    else:
        clock, increment = timeIncrement, "0"

    try:
        clock = int(clock)
    except ValueError:
        clock = 60

    try:
        increment = int(increment)
    except ValueError:
        increment = 0

    return clock, increment

class RatingModel:
    def __init__(self, time_increment="60+0"):
        self.device = get_device()
        self.model = self.load_model(f"models/best_model_{time_increment}.pth")
        self.clock, _ = parse_clock(time_increment)
        self.rating_range = increment_rating_ranges[time_increment]
        self.hidden_state = None
        self.history = []

    def load_model(self, model_path):
        model = ChessEloPredictor(training=False)
        model_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(model_dict["model_state_dict"])
        model.eval()

        return model

    def estimate_rating(self, board, white_clock):
        board_array = board_to_array(board).to(self.device)
        white_clock = white_clock[0] / self.clock
        white_clock = torch.tensor(white_clock, dtype=torch.float).to(self.device)

        with torch.no_grad():
            outputs, last, self.hidden_state = self.model(board_array.unsqueeze(0).unsqueeze(0), white_clock.unsqueeze(0).unsqueeze(0), torch.tensor([1]), self.hidden_state)
            prediction = last.squeeze().cpu().numpy()

        rating_range = self.rating_range[1] - self.rating_range[0]
        prediction = prediction * rating_range + self.rating_range[0]

        return prediction



