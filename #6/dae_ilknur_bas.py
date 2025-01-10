import torch
from torch import Tensor, cuda, rand
from torch.nn import Module, GRU, Linear, ReLU


class MySystem(Module):

    def __init__(self):
        super().__init__()

        # out_all, out_last = (take out_all)
        # 128 output features
        self.bgru1 = GRU(input_size=1025, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        # Add two more Bi-GRU layers having 256 input and 128 output feature
        self.bgru2 = GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.bgru3 = GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        # Add one GRU layer having 256 input and 128 output features.
        self.gru = GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.linear = Linear(in_features=128, out_features=1025)
        # returns 0 if it receives any negative input, but for any positive value x it returns that value back.
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        h = x  # if x.ndimension() == 4 else x.unsqueeze(1)
        h = torch.transpose(h, 1, 2)
        # print("after transpose", h.shape)  # torch.Size([4, 60, 1025])
        h, _ = self.bgru1(h)
        # print("after bgru1", h.shape)  # torch.Size([4, 60, 2])
        h, _ = self.bgru2(h)
        # print("after bgru2", h.shape)
        h, _ = self.bgru3(h)
        # print("after bgru3", h.shape)
        h, _ = self.gru(h)
        # print("after gru", h.shape)
        h = self.linear(h)
        # print("after linear", h.shape)
        h = self.relu(h)
        # print("after relu", h.shape)
        h = torch.transpose(h, 2, 1)
        # print("after everything", h.shape)
        return h


def main():
    device = 'cuda' if cuda.is_available() else 'cpu'

    model = MySystem()
    model = model.to(device)

    batch_size = 4
    d_time = 60
    # ...
    x = rand(batch_size, 1025, d_time) # (batch_size, 1025, d_time)
    y = rand(batch_size, 1025, d_time)  # (batch_size, d_time, 1025)

    # Give them to the appropriate device.
    x = x.to(device)
    y = y.to(device)

    print("x.shape", x.shape)
    print("y.shape", y.shape)

    # Get the predictions .
    y_hat = model(x)
    print("y_hat.shape", y_hat.shape)  # ([4, 1025, 60])
    # print("y.shape", y.shape)
    print("y_hat", y_hat)


if __name__ == '__main__':
    main()

# EOF
