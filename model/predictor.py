import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class ChessEloPredictor(nn.Module):
    # RatingNet
    def __init__(self, conv_filters=16, lstm_layers=2, dropout_rate=0.5, lstm_h=64, fc1_h=16, bidirectional=False):
        super(ChessEloPredictor, self).__init__()
        self.conv1 = nn.Conv2d(12, conv_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_filters)
        self.conv2 = nn.Conv2d(conv_filters, conv_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_filters * 2)
        self.conv3 = nn.Conv2d(conv_filters * 2, conv_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_filters * 4)
        self.conv4 = nn.Conv2d(conv_filters * 4, conv_filters * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_filters * 8)
        self.pool = nn.AvgPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=conv_filters * 8 + 1, hidden_size=lstm_h, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.fc1 = nn.Linear(lstm_h, fc1_h)
        if bidirectional:
            self.fc1 = nn.Linear(lstm_h * 2, fc1_h)
        self.fc2 = nn.Linear(fc1_h, 2)

    def forward(self, positions, clocks, lengths):
        # CNN-LSTM model

        batch_size = positions.size(0)
        sequence_length = positions.size(1)
        positions = positions.view(-1, 12, 8, 8)
        x = F.leaky_relu(self.bn1(self.conv1(positions)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = self.dropout1(x)
        x = x.view(batch_size, sequence_length, -1)
        # [batch=32, sequence_length, hidden=256]
        clocks = clocks.unsqueeze(2)
        # [batch, sequence_length, 1]
        lstm_input = torch.cat((x, clocks), dim=2)
        # [batch=32, sequence_length, hidden=258]
        packed_input = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Take the last time step
        # lstm_output = lstm_output[torch.arange(lstm_output.size(0)), lengths - 1]
        # [batch, hidden_dim=128]
        y = F.leaky_relu(self.fc1(lstm_output))
        y = self.dropout1(y)
        y = self.fc2(y)

        # Use torch.arange to select the last time step for each sequence
        idx = torch.arange(batch_size)
        last_time_step_output = y[idx, lengths - 1, :]
        return y, last_time_step_output


def train_one_epoch(model, train_loader, device, criterion, optimizer, ratings_mean=1650, ratings_std=430):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        positions = batch['positions'].to(device)
        clocks = batch['clocks'].to(device)
        targets = batch['targets'].to(device)
        lengths = batch['lengths']
        optimizer.zero_grad()
        all, outputs = model(positions, clocks, lengths)
        loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    return total_train_loss / len(train_loader)


def validate(model, val_loader, device, criterion, ratings_mean=1650, ratings_std=430):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            positions = batch['positions'].to(device)
            clocks = batch['clocks'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['lengths']
            all, outputs = model(positions, clocks, lengths)
            loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)
            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)


def mae_per_item(outputs, targets, ratings_mean, ratings_std):
    outputs_rescaled = outputs * ratings_std + ratings_mean
    targets_rescaled = targets * ratings_std + ratings_mean
    mae = torch.abs(outputs_rescaled - targets_rescaled)
    return mae.mean(dim=1)  # Mean absolute error per item


def mae_per_color(outputs, targets, ratings_mean, ratings_std):
    outputs_rescaled = outputs * ratings_std + ratings_mean
    targets_rescaled = targets * ratings_std + ratings_mean
    mae = torch.abs(outputs_rescaled - targets_rescaled)
    white_mae = mae[:, 0]
    black_mae = mae[:, 1]
    return white_mae, black_mae  # Mean absolute error per color


def mse_per_item(outputs, targets, ratings_mean, ratings_std):
    outputs_rescaled = outputs * ratings_std + ratings_mean
    targets_rescaled = targets * ratings_std + ratings_mean
    mse = (outputs_rescaled - targets_rescaled) ** 2
    return mse.mean(dim=1)  # Mean squared error per item


def mse_per_color(outputs, targets, ratings_mean, ratings_std):
    outputs_rescaled = outputs * ratings_std + ratings_mean
    targets_rescaled = targets * ratings_std + ratings_mean
    mse = (outputs_rescaled - targets_rescaled) ** 2
    white_mse = mse[:, 0]
    black_mse = mse[:, 1]
    return white_mse, black_mse  # Mean squared error per color


def test(model, test_loader, device, criterion, ratings_mean=1650, ratings_std=430):
    model.eval()
    total_test_loss = 0
    loss_by_time_control = {'ultrabullet': 0, 'bullet': 0, 'blitz': 0, 'rapid': 0, 'classical': 0}
    count_by_time_control = {'ultrabullet': 0, 'bullet': 0, 'blitz': 0, 'rapid': 0, 'classical': 0}
    total_white_loss = 0
    total_black_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            positions = batch['positions'].to(device)
            clocks = batch['clocks'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['lengths']
            time_controls = batch['time_controls']
            game_results = batch['results']  # Assuming results are part of the batch

            all, outputs = model(positions, clocks, lengths)
            # Test setting outputs to mean rating
            # outputs = torch.zeros_like(outputs)
            loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)

            total_test_loss += loss.item()
            if str(criterion) == "L1Loss()":
                print("Using MAE")
                mae = mae_per_item(outputs, targets, ratings_mean, ratings_std)
                black, white = mae_per_color(outputs, targets, ratings_mean, ratings_std)
                total_white_loss += white.mean().item()
                total_black_loss += black.mean().item()
                for idx, time_control in enumerate(time_controls):
                    loss_by_time_control[time_control] += mae[idx].item()
                    count_by_time_control[time_control] += 1
            elif str(criterion) == "MSELoss()":
                print("Using MSE")
                mse = mse_per_item(outputs, targets, ratings_mean, ratings_std)
                black, white = mse_per_color(outputs, targets, ratings_mean, ratings_std)
                total_white_loss += white.sum().item()
                total_black_loss += black.sum().item()
                for idx, time_control in enumerate(time_controls):
                    loss_by_time_control[time_control] += mse[idx].item()
                    count_by_time_control[time_control] += 1
            else:
                print("Error, Unknown loss function")

    for key in loss_by_time_control:
        if count_by_time_control[key] > 0:
            loss_by_time_control[key] /= count_by_time_control[key]

    return total_test_loss / len(test_loader), loss_by_time_control, total_white_loss, total_black_loss
