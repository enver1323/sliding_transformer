import torch
from random import randint
from torch import optim, nn
from tqdm import tqdm

from config import PollutionConfig, StockConfig
from model import SlidingTransformer, PollutionPredictor
from utils import compute_path_increments

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, x_train, y_train, epochs: int, clip_value: float = None):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    n = x_train.size(0)
    progress_bar = tqdm(range(epochs))

    for epoch in progress_bar:
        total_loss = 0
        for step in range(n):
            optimizer.zero_grad()

            idx = randint(0, n - 1)

            x = compute_path_increments(x_train[idx])
            y = compute_path_increments(y_train[idx])

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()

            loss.backward()

            if clip_value is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

            progress_bar.set_description_str(
                f"[Epoch {epoch} | Step {step + 1}] Loss: {loss.item():.2f}, Cum loss: {(total_loss / (step + 1)):.2f}")

        print(f"[Epoch {epoch}] Cum loss: {(total_loss / n):.2f}")

    return model


def app():
    config = PollutionConfig(device=DEVICE)
    x_train, y_train = config.preprocess_dataset("data/pollution/train.csv")
    # config = StockConfig(device=DEVICE)
    # x_train, y_train = config.preprocess_dataset("data/msft/msft.csv")

    print(x_train.shape, y_train.shape)
    print("Training ...")

    sliding_transformer = SlidingTransformer(**config.model_params).to(DEVICE)
    transformer_model = PollutionPredictor(sliding_transformer, config.horizon).to(DEVICE)
    train(model=transformer_model, x_train=x_train, y_train=y_train, epochs=10)


if __name__ == '__main__':
    app()
