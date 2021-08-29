import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.reshape(torch.tensor([0.5], requires_grad=True), (1, -1)))
        self.b = torch.nn.Parameter(torch.reshape(torch.tensor([0.5], requires_grad=True), (1, -1)))

    def forward(self, x):
        return x @ self.w + self.b


def main():
    data = pd.read_csv("length_weight.csv")
    x_train = torch.reshape(torch.tensor(data.pop("length").to_numpy(), dtype=torch.float), (-1, 1))
    y_train = torch.reshape(torch.tensor(data.pop("weight").to_numpy(), dtype=torch.float), (-1, 1))
    learning_rate = 0.0001
    num_epoch = 500000

    model = Model()
    loss_fun = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD([model.w, model.b], lr=learning_rate)

    for epoch in range(num_epoch):
        y_hat = model(x_train)
        loss = loss_fun(y_train, y_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(model.state_dict())
    print(f"Loss: {loss_fun(y_train, y_hat)}")

    plt.title("Linear regression 2d")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plots the actual values
    plt.plot(x_train.detach(), y_train.detach(), 'o', label='$(x^{(i)},y^{(i)})$', color='blue')
    plt.plot(x_train.detach(), model.forward(x_train).detach(), '-', color='red', label='$\\hat y = f(x) = xW+b$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
