import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
from mpl_toolkits.mplot3d import axes3d, art3d


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.reshape(torch.tensor([0.0, 0.0], requires_grad=True), (2, -1)))
        self.b = torch.nn.Parameter(torch.reshape(torch.tensor([0.0], requires_grad=True), (1, -1)))

    def forward(self, x):
        return x @ self.w + self.b


def main():
    # Setting up test data
    data = pd.read_csv("day_length_weight.csv")
    y_train = torch.reshape(torch.tensor(data.pop("day").to_numpy(), dtype=torch.float), (-1, 1))
    x_train = torch.reshape(torch.tensor(data.to_numpy(), dtype=torch.float), (-1, 2))

    # Setting hyper variables
    learning_rate = 0.00001
    num_epoch = 100000

    # Initializing model, loss and optimizer
    model = Model()
    loss_fun = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD([model.w, model.b], lr=learning_rate)

    # Optimizing model
    for epoch in range(num_epoch):
        y_hat = model(x_train)
        loss = loss_fun(y_train, y_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Printing result
    print(model.state_dict())
    print(f"Loss: {loss_fun(y_train, y_hat)}")

    # Plotting result
    fig = plt.figure("Linear regression 3d")
    ax = fig.add_subplot(projection='3d')
    ax.set_title("Linear regression 3d")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    # Scatter for the actual data
    ax.scatter(x_train[:, 0].detach().numpy(), x_train[:, 1].detach().numpy(), y_train[:, 0].detach().numpy(),
               label='$(x_1^{(i)},x_2^{(i)},y^{(i)})$', color='blue')

    # Plotting plane for the result function
    x1_grid, x2_grid = np.meshgrid(
        np.arange(start=int(torch.min(x_train[:, 0]).item()), stop=int(torch.max(x_train[:, 0]).item() + 2), step=1.0,
                  dtype='float32'),
        np.arange(start=int(torch.min(x_train[:, 1]).item()), stop=int(torch.max(x_train[:, 1]).item() + 2), step=1.0,
                  dtype='float32'))
    y_grid = np.empty([x1_grid.shape[0], x1_grid.shape[1]])
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.forward(torch.reshape(torch.tensor([x1_grid[i, j], x2_grid[i, j]]), (1, 2)))

    # Because of bug in matplotlib facecolors2d and edgecolors2d have to be defined to be able to add the legend
    surface = ax.plot_surface(x1_grid, x2_grid, y_grid, color="red", label="$\\hat y = f(x1,x2) = x*W+b$")
    surface._facecolors2d = surface._facecolor3d
    surface._edgecolors2d = surface._edgecolor3d
    ax.legend()

    # Showing plot

    plt.show()


if __name__ == '__main__':
    main()
