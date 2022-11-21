import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import StockConfig, ETTConfig, PollutionConfig, Dataset_ETT_hour
from model import SlidingTransformer, SlidingAttention
from utils.preprocessing import compute_path_increments

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(
    model,
    data_loader,
    epochs: int,
    batch_size=64,
    num_steps=20000,
    clip_value: float = None,
):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(enumerate(data_loader))
        for step, (x,y, _, _) in progress_bar:
            optimizer.zero_grad()

            x = x.to(DEVICE).type(torch.float32)
            y = y.to(DEVICE).type(torch.float32)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()

            loss.backward()

            if clip_value is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

            progress_bar.set_description_str(
                f"[Epoch {epoch} | Step {step + 1}] Loss: {loss.item():.4f}, Cum loss: {(total_loss / (step + 1)):.4f}")

    return model


class SlidingPredictor(nn.Module):
    def __init__(self, sliding_model, d_model, out_dim):
        super(SlidingPredictor, self).__init__()
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=8, kernel_size=3, stride=1)  # (batch, 16, 13)
        self.max_pool = nn.MaxPool1d(2, 1)

        self.sliding_net1 = sliding_model(
            c_in=8,
            d_model=16,
            kernel=32,
            stride=1,
            out_dim=32,
            num_heads=4,
            dropout=0.1,
        )  # (batch, 32, 16)

        self.linear1 = nn.Linear(32, 1)  # (batch, 1, 16)
        self.linear2 = nn.Linear(32, 16)  # (batch, 1, 16)
        self.out = nn.Linear(16, out_dim)  # (batch, 1, out_dim)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.conv(x)
        x = self.max_pool(x)
        x = nn.functional.relu(x)

        x = x.transpose(2, 1)

        x = self.sliding_net1(x)

        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = x.transpose(2, 1)

        x = self.linear2(x)
        x = nn.functional.relu(x)

        x = self.out(x)
        return x.transpose(-1, -2)


def app():
    # config = PollutionConfig(device=DEVICE)
    # x_train, y_train = config.preprocess_dataset("data/pollution/train.csv")
    # config = StockConfig(device=DEVICE, horizon=1)
    # x_train, y_train = config.preprocess_dataset("data/stock/tesla.csv", '%M/%d/%Y')

    # config = ETTConfig(
    #     in_features=['OT'],
    #     lookback_window=336,
    #     horizon=720,
    #     device=DEVICE,
    # )
    # x_train, y_train = config.preprocess_dataset("data/ETDataset/ETT-small/ETTh1.csv")

    config = {
        "in_features": 1,
        "out_features": 1,
    }

    dataset = Dataset_ETT_hour(
        root_path="data/ETDataset/ETT-small",
        size=[336, 336, 720],
    )
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print("Training ...")

    sliding_model = SlidingAttention
    model = SlidingPredictor(sliding_model, 1, 720).to(DEVICE)

    train(
        model=model,
        data_loader=data_loader,
        epochs=10,
        batch_size=32,
    )


if __name__ == '__main__':
    app()
