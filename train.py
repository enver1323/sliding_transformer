import torch
from torch import optim, nn
from tqdm import tqdm

from config import StockConfig
from model import SlidingTransformer, SlidingAttention
from utils import compute_path_increments

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(
    model,
    x_train,
    y_train,
    epochs: int,
    batch_size=64,
    num_steps=20000,
    clip_value: float = None,
    compute_increments: bool = False
):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    progress_bar = tqdm(range(epochs))

    if compute_increments:
        x_train = compute_path_increments(x_train)
        y_train = compute_path_increments(y_train)

    for epoch in progress_bar:
        total_loss = 0
        for step in range(num_steps):
            optimizer.zero_grad()

            indices = torch.randint(len(x_train), (batch_size,))

            x = x_train[indices]
            y = y_train[indices]

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()

            loss.backward()

            if clip_value is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()

            progress_bar.set_description_str(
                f"[Epoch {epoch} | Step {step + 1}] Loss: {loss.item():.2f}, Cum loss: {(total_loss / (step + 1)):.2f}")

        print(f"[Epoch {epoch}] Cum loss: {(total_loss / num_steps):.2f}")

    return model


class SlidingPredictor(nn.Module):
    def __init__(self, sliding_model, d_model, out_dim):
        super(SlidingPredictor, self).__init__()
        # self.conv = nn.LazyConv1d(base_model.d_model, conv_kernel)
        self.sliding_net1 = sliding_model(
            d_model=d_model,
            kernel=32,
            stride=1,
            out_dim=64,
            num_heads=4,
            dropout=0.2,
        )  # (batch, 32, 8)

        # self.conv1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=8, stride=2)  # (batch, 16, 13)
        # self.avg_pool1 = nn.MaxPool1d(2)  # (batch, 16, 6)
        # self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2)  # (batch, 32, 2)
        # self.avg_pool2 = nn.MaxPool1d(2)  # (batch, 32, 1)

        self.sliding_net2 = sliding_model(
            d_model=64,
            kernel=16,
            stride=1,
            num_heads=4,
            dropout=0.1
        )

        self.linear1 = nn.Linear(16, 1)  # (batch, 1, 16)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, out_dim)  # (batch, 1, out_dim)

    def forward(self, x):
        x = self.sliding_net1(x)

        # x = x.transpose(2, 1)
        # x = self.conv1(x)
        # x = self.avg_pool1(x)
        # x = nn.functional.relu(x)
        # x = self.conv2(x)
        # x = self.avg_pool2(x)
        # x = nn.functional.relu(x)
        # x = x.transpose(2, 1)

        x = self.sliding_net2(x)
        x = x.transpose(2, 1)

        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = x.transpose(2, 1)


        x = self.linear2(x)
        x = nn.functional.relu(x)
        x = self.linear3(x)
        x = nn.functional.relu(x)
        x = self.out(x)
        return x


class LSTMBaseline(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super(LSTMBaseline, self).__init__()
        self.lstm1 = nn.LSTM(embed_dim, 50, batch_first=True)
        self.lstm2 = nn.LSTM(50, 64, batch_first=True)
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, out_dim)

    def forward(self, x):
        x, _ = self.lstm1(x)
        _, (x, _) = self.lstm2(x)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        x = self.out(x)
        return x


def app():
    # config = PollutionConfig(device=DEVICE)
    # x_train, y_train = config.preprocess_dataset("data/pollution/train.csv")
    config = StockConfig(device=DEVICE, horizon=1)
    x_train, y_train = config.preprocess_dataset("data/stock/tesla.csv", '%M/%d/%Y')

    print(x_train.shape, y_train.shape)
    print("Training ...")

    sliding_model = SlidingAttention
    # sliding_model = SlidingTransformer
    model = SlidingPredictor(sliding_model, len(config.in_features), config.horizon).to(DEVICE)
    # model = LSTMBaseline(len(config.in_features), config.horizon).to(DEVICE)
    train(
        model=model,
        x_train=x_train,
        y_train=y_train,
        epochs=10,
        batch_size=32,
        compute_increments=False
    )


if __name__ == '__main__':
    app()
