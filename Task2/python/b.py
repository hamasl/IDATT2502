import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
import math


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.reshape(torch.tensor([0.0, 0.0], dtype=float, requires_grad=True), (2, -1)))
        self.b = torch.nn.Parameter(torch.reshape(torch.tensor([0.0], dtype=float, requires_grad=True), (1, -1)))

    def logits(self, x):
        return x @ self.w + self.b

    def forward(self, x):
        return torch.sigmoid(self.logits(x))


def main():
    x_train = torch.reshape(torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float), (-1, 2))
    y_train = torch.reshape(torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=float), (-1, 1))
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

    # Plotting result
    fig = plt.figure("NAND function")
    ax = fig.add_subplot(projection='3d')
    ax.set_title("NAND function")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    # Scatter for the actual data
    ax.scatter(x_train[:, 0].detach().numpy(), x_train[:, 1].detach().numpy(), y_train[:, 0].detach().numpy(),
               label='$(x_1^{(i)},x_2^{(i)},y^{(i)})$', color='blue')

    # Plotting plane for the result function
    x1_grid, x2_grid = np.meshgrid(
        np.arange(start=torch.min(x_train[:, 0]).item(), stop=torch.max(x_train[:, 0]).item(), step=0.01,
                  dtype='float64'),
        np.arange(start=torch.min(x_train[:, 1]).item(), stop=torch.max(x_train[:, 1]).item(), step=0.01,
                  dtype='float64'))
    y_grid = np.empty([x1_grid.shape[0], x1_grid.shape[1]])
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.forward(torch.reshape(torch.tensor([x1_grid[i, j], x2_grid[i, j]]), (1, 2)))

    # Because of bug in matplotlib facecolors2d and edgecolors2d have to be defined to be able to add the legend
    surface = ax.plot_wireframe(x1_grid, x2_grid, y_grid, color="red", label="$\\hat y = f(x1,x2) = sigmoid(x*W+b)$", rstride=5, cstride=5)
    #surface._facecolors2d = surface._facecolor3d
    #surface._edgecolors2d = surface._edgecolor3d
    ax.legend()

    # Showing plot

    plt.show()


if __name__ == '__main__':
    main()