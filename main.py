import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import time
import wandb

from data.dataset import ChessGamesDataset, collate_fn
from model.predictor import ChessEloPredictor, train_one_epoch, validate, test, WeightedMSELoss
from util import get_device

def main(data_dir, experiment_name, train, criterion, epochs):
    i = 1
    while os.path.exists(f"runs/{experiment_name}_{i}"):
        i += 1
    # best_path = "models/2024-09_games_60+0/model_3.pth"
    # Weighted loss
    # For a game, the weight of the loss is greater the later in the game it is
    # criterion = WeightedMSELoss()

    if criterion == "MSELoss":
        criterion = nn.MSELoss()
    elif criterion == "L1Loss":
        criterion = nn.L1Loss()
    elif criterion == "WeightedMSELoss":
        criterion = WeightedMSELoss()


    params = {
        'train_batch_size': 32,
        'val_batch_size': 1024,
        'num_workers': 4,
        'learning_rate': 0.0001,
        'weight_decay': 1e-5,
        'epochs': epochs,
        'optimizer': 'Adam',
        'patience': 5,
        'lr_factor': 0.5,
        "conv_filters": 32,
        "lstm_layers": 3,
        "bidirectional": True,
        "dropout_rate": 0.5,
        "lstm_h": 64,
        "fc1_h": 32
    }

    wandb.init(project='chess-elo-prediction', name=experiment_name, config=params)

    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
    model_dir = os.path.join('models', experiment_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join('runs', experiment_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Split into train, val, and test sets
    train_val_files, test_files = train_test_split(all_files, test_size=0.1, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)

    train_dataset = ChessGamesDataset(train_files)
    val_dataset = ChessGamesDataset(val_files)
    test_dataset = ChessGamesDataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=params["train_batch_size"], shuffle=True, collate_fn=collate_fn,
                              num_workers=params['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=False, collate_fn=collate_fn,
                            num_workers=params['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=params['val_batch_size'], shuffle=False, collate_fn=collate_fn,
                             num_workers=params['num_workers'])

    device = get_device()
    print("Device: ", device)
    model = ChessEloPredictor(params["conv_filters"], params["lstm_layers"], params["dropout_rate"],
                              params["lstm_h"], params["fc1_h"], params["bidirectional"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=params['patience'], factor=params['lr_factor'])
    best_val_loss = float('inf')
    best_epoch = 0
    start = time.time()

    if train:
        print("Training model")
        for epoch in range(params['epochs']):
            epoch_start = time.time()
            train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
            print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}')
            val_loss = validate(model, val_loader, device, nn.L1Loss())
            print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            epoch_duration = (time.time() - epoch_start) / 60
            writer.add_scalar('Timing/Epoch Duration', epoch_duration, epoch)

            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch_duration": epoch_duration})

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_path = os.path.join(model_dir, f'model_{epoch + 1}.pth')
                torch.save({'model_state_dict': model.state_dict(),
                            'params': params},
                           best_path)
                print("Saved best model")
        end = time.time()
        print("Training duration: ", (end - start) / 60)
        print("best val loss: ", best_val_loss)
        print("best val epoch: ", best_epoch)
        writer.close()

    print("Testing model")

    saved_model = torch.load(best_path, map_location=get_device())
    model.load_state_dict(saved_model["model_state_dict"])
    test_loss = test(model, test_loader, device, criterion)
    end = time.time()
    print("seconds: ", round(end - start))
    print("Test Loss: ", test_loss)
    return 1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--data_dir", type=str, default="data/unpacked_games")
    argparser.add_argument("--experiment_name", type=str, default="test")
    argparser.add_argument("--train", type=bool, default=True)
    argparser.add_argument("--criterion", type=str, default="MSELoss")
    argparser.add_argument("--epochs", type=int, default=50)

    args = argparser.parse_args()

    main(args.data_dir, args.experiment_name, args.train, args.criterion, args.epochs)