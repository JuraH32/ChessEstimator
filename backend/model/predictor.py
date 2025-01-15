import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class ChessEloPredictor(nn.Module):
    # RatingNet
    def __init__(self, conv_filters=32, lstm_layers=3, dropout_rate=0.5, lstm_h=64, fc1_h=32, bidirectional=True, training=True):
        super(ChessEloPredictor, self).__init__()
        self.training = training
        self.conv1 = nn.Conv2d(12, conv_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_filters)
        self.conv2 = nn.Conv2d(conv_filters, conv_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_filters * 2)
        self.conv3 = nn.Conv2d(conv_filters * 2, conv_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_filters * 4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(input_size=conv_filters * 4 + 1, hidden_size=lstm_h, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bidirectional)

        lstm_output_dim = lstm_h * 2 if bidirectional else lstm_h

        self.fc1 = nn.Linear(lstm_output_dim, fc1_h)
        self.fc2 = nn.Linear(fc1_h, 1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, positions, clocks, lengths, hidden_state=None):
        # CNN-LSTM model
        batch_size = positions.size(0)
        seq_len = positions.size(1)
        positions = positions.view(-1, 12, 8, 8)

        x = F.leaky_relu(self.bn1(self.conv1(positions)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(batch_size, seq_len, -1)

        # Combine CNN features with clock input
        clocks = clocks.unsqueeze(2)  # [batch_size, seq_len, 1]
        lstm_input = torch.cat((x, clocks), dim=2)  # [batch_size, seq_len, conv_filters * 4 + 1]

        new_states = None

        if self.training:
            # Pack padded sequence for LSTM
            packed_input = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input)
            lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_output, new_states = self.lstm(lstm_input, hidden_state)

        # Fully connected layers
        lstm_output = F.leaky_relu(self.fc1(lstm_output))
        lstm_output = self.dropout2(lstm_output)
        rating_estimates = self.fc2(lstm_output)  # [batch_size, seq_len, 1]

        # Make the rating estimates a value between 0 and 1
        rating_estimates = torch.sigmoid(rating_estimates)

        # Final prediction: Take the output corresponding to the last valid time step for each sequence
        idx = torch.arange(batch_size)
        final_predictions = rating_estimates[idx, lengths - 1, :]  # [batch_size, 1]

        return rating_estimates, final_predictions, new_states


# Class for weighted loss
class WeightedMSELoss(nn.Module):
    def __init__(self, max_moves=100):
        super(WeightedMSELoss, self).__init__()
        self.weights = [torch.arange(1, i + 1, dtype=torch.float) / i for i in range(1, max_moves + 1)]

    def forward(self, predictions, targets, game_lengths):
        """
        Compute the weighted MSE loss.

        Args:
            predictions (torch.Tensor): Predicted values of shape [batch_size, predictions, seq_len].
            targets (torch.Tensor): Ground truth values of shape [batch_size, seq_len].

        Returns:
            torch.Tensor: The weighted MSE loss.
        """
        batch_size, length, seq_len = predictions.shape
        device = predictions.device

        max_length = max(game_lengths)

        # Move weights to device and gather relevant weights for each batch
        weight_tensors = torch.stack([F.pad(self.weights[l - 1].to(device), (0, max_length - l)) for l in game_lengths])  # Shape: [batch_size, max_game_length]

        # Expand weights to match the prediction dimensions
        weight_tensors = F.pad(weight_tensors, (0, seq_len - weight_tensors.size(1)))  # Pad to seq_len
        weight_tensors = weight_tensors.unsqueeze(2).expand(-1, -1, length)  # Shape: [batch_size, seq_len, length]

        # Adjust predictions to shape [batch_size, seq_len, predictions]
        predictions = predictions.permute(0, 2, 1)

        # Expand targets to match predictions shape: [batch_size, seq_len, predictions]
        targets = targets.unsqueeze(2).expand(-1, -1, length)

        # Compute the squared error
        squared_error = (predictions - targets) ** 2

        # Apply weights to the squared error
        weighted_error = squared_error * weight_tensors

        # Compute weighted mean squared error for each sequence
        weight_sums = weight_tensors.sum(dim=1)  # Sum of weights for each batch
        weighted_loss = weighted_error.sum(dim=1) / weight_sums  # Weighted loss for each batch

        # Total loss across the batch
        loss = weighted_loss.sum()

        return loss

def train_one_epoch(model, train_loader, device, criterion, optimizer, ratings_min=900, ratings_max=2400):
    model.train()
    total_train_loss = 0
    rating_range = ratings_max - ratings_min
    for batch in train_loader:
        positions = batch['positions'].to(device)
        clocks = batch['clocks'].to(device)
        targets = batch['targets'].to(device)
        lengths = batch['lengths']
        optimizer.zero_grad()
        all, outputs, _ = model(positions, clocks, lengths)
        if criterion.__class__.__name__ == "WeightedMSELoss":
            loss = criterion(all * rating_range + ratings_min, targets * rating_range + ratings_min, lengths)
        else:
            loss = criterion(outputs * rating_range + ratings_min, targets * rating_range + ratings_min)

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    return total_train_loss / len(train_loader)


def validate(model, val_loader, device, criterion, ratings_min=900, ratings_max=2400):
    model.eval()
    total_val_loss = 0
    ratings_range = ratings_max - ratings_min
    with torch.no_grad():
        for batch in val_loader:
            positions = batch['positions'].to(device)
            clocks = batch['clocks'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['lengths']
            all, outputs, _ = model(positions, clocks, lengths)
            loss = criterion(outputs * ratings_range + ratings_min, targets * ratings_range + ratings_min)
            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)


def mae_per_item(outputs, targets, ratings_min, ratings_max):
    ratings_range = ratings_max - ratings_min
    outputs_rescaled = outputs * ratings_range + ratings_min
    targets_rescaled = targets * ratings_range + ratings_min
    mae = torch.abs(outputs_rescaled - targets_rescaled)
    return mae.mean(dim=1)  # Mean absolute error per item


def mae_per_color(outputs, targets, ratings_min, ratings_max):
    ratings_range = ratings_max - ratings_min
    outputs_rescaled = outputs * ratings_range + ratings_min
    targets_rescaled = targets * ratings_range + ratings_min
    mae = torch.abs(outputs_rescaled - targets_rescaled)
    white_mae = mae[:, 0]
    black_mae = mae[:, 1]
    return white_mae, black_mae  # Mean absolute error per color


def mse_per_item(outputs, targets, ratings_min, ratings_max):
    ratings_range = ratings_max - ratings_min
    outputs_rescaled = outputs * ratings_range + ratings_min
    targets_rescaled = targets * ratings_range + ratings_min
    mse = (outputs_rescaled - targets_rescaled) ** 2
    return mse.mean(dim=1)  # Mean squared error per item


def mse_per_color(outputs, targets, ratings_min, ratings_max):
    ratings_range = ratings_max - ratings_min
    outputs_rescaled = outputs * ratings_range + ratings_min
    targets_rescaled = targets * ratings_range + ratings_min
    mse = (outputs_rescaled - targets_rescaled) ** 2
    white_mse = mse[:, 0]
    black_mse = mse[:, 1]
    return white_mse, black_mse  # Mean squared error per color


def test(model, test_loader, device, criterion, ratings_min=900, ratings_max=2400):
    model.eval()
    total_test_loss = 0
    loss_by_time_control = {'ultrabullet': 0, 'bullet': 0, 'blitz': 0, 'rapid': 0, 'classical': 0}
    count_by_time_control = {'ultrabullet': 0, 'bullet': 0, 'blitz': 0, 'rapid': 0, 'classical': 0}
    total_white_loss = 0
    total_black_loss = 0

    ratings_range = ratings_max - ratings_min

    with torch.no_grad():
        for batch in test_loader:
            positions = batch['positions'].to(device)
            clocks = batch['clocks'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['lengths']
            time_controls = batch['time_controls']
            game_results = batch['results']  # Assuming results are part of the batch

            all, outputs, _ = model(positions, clocks, lengths)
            # Test setting outputs to mean rating
            # outputs = torch.zeros_like(outputs)
            loss = criterion(outputs * ratings_range + ratings_min, targets * ratings_range + ratings_min)

            total_test_loss += loss.item()
            if str(criterion) == "L1Loss()":
                print("Using MAE")
                mae = mae_per_item(outputs, targets, ratings_min, ratings_max)
                # black, white = mae_per_color(outputs, targets, ratings_mean, ratings_std)
                # total_white_loss += white.mean().item()
                # total_black_loss += black.mean().item()
                for idx, time_control in enumerate(time_controls):
                    loss_by_time_control[time_control] += mae[idx].item()
                    count_by_time_control[time_control] += 1
            elif str(criterion) == "MSELoss()":
                print("Using MSE")
                mse = mse_per_item(outputs, targets, ratings_min, ratings_max)
                # black, white = mse_per_color(outputs, targets, ratings_mean, ratings_std)
                # total_white_loss += white.sum().item()
                # total_black_loss += black.sum().item()
                for idx, time_control in enumerate(time_controls):
                    loss_by_time_control[time_control] += mse[idx].item()
                    count_by_time_control[time_control] += 1
            else:
                print("Error, Unknown loss function")

    for key in loss_by_time_control:
        if count_by_time_control[key] > 0:
            loss_by_time_control[key] /= count_by_time_control[key]

    return total_test_loss / len(test_loader), loss_by_time_control, total_white_loss, total_black_loss
