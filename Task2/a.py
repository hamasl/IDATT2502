import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
import math


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.reshape(torch.tensor([0.0], dtype=float, requires_grad=True), (1, -1)))
        self.b = torch.nn.Parameter(torch.reshape(torch.tensor([0.0], dtype=float, requires_grad=True), (1, -1)))

    def logits(self, x):
        return x @ self.w + self.b

    def forward(self, x):
        return torch.sigmoid(self.logits(x))


def main():
    x_train = torch.reshape(torch.tensor([0.0,1.0], dtype=float), (-1, 1))
    y_train = torch.reshape(torch.tensor([1.0,0.0], dtype=float), (-1, 1))
    learning_rate = 0.01
    num_epoch = 100000

    model = Model()
    optimizer = torch.optim.SGD([model.w, model.b], lr=learning_rate)

    for epoch in range(num_epoch):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model.logits(x_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(model.state_dict())
    print(f"Loss: {loss}")

    plt.title("NOT function")
    plt.xlabel("x")
    plt.ylabel("y")

    # Plots the actual values
    plt.plot(x_train.detach(), y_train.detach(), 'o', label='$(x^{(i)},y^{(i)})$', color='blue')
    # Creating numpy array ranging from smallest to largest entry in the tensor
    x = np.arange(start=torch.min(x_train).item(), stop=torch.max(x_train).item(), step=0.01,
                  dtype='float64')
    # Converting x to a tensor and running forward to get y_hat, then detaching it so it can be displayed
    y_hat_plot = model.forward(torch.reshape(torch.tensor(x), (-1, 1))).detach()
    plt.plot(x, y_hat_plot, '-', color='red', label='$\\hat y = f(x) = sigmoid(xW+b)$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()