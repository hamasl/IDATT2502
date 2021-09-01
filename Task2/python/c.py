import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
import math


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.min_w_value = -1
        self.max_w_value = 1
        # TODO remove
        # torch.rand uses values [0,1)
        self.w1 = torch.rand((2, 2), dtype=float, requires_grad=True)
        #self.w1 = (self.max_w_value-self.min_w_value)*torch.rand((2, 2), dtype=float, requires_grad=True)+self.min_w_value
        #self.w1 = 2*torch.rand((2, 2), dtype=float, requires_grad=True)-torch.ones((2,2))
        self.b1 = torch.rand((1, 2), dtype=float, requires_grad=True)
        self.w2 = torch.rand((2, 1), dtype=float, requires_grad=True)
        #self.w2 = (self.max_w_value-self.min_w_value)*torch.rand((2, 1), dtype=float, requires_grad=True)+self.min_w_value
        #self.w2 = 2*torch.rand((2, 1), dtype=float, requires_grad=True)-torch.ones((2,1))
        self.b2 = torch.rand((1, 1), dtype=float, requires_grad=True)

    def get_params(self):
        print(self.w1, self.b1, self.w2, self.b2)
        return [self.w1, self.b1, self.w2, self.b2]

    def f1(self, x):
        return torch.sigmoid(x @ self.w1 + self.b1)

    def f2(self, h):
        return torch.sigmoid(h @ self.w2 + self.b2)

    def logits(self, h):
        return h @ self.w2 + self.b2

    def forward(self, x):
        return self.f2(self.f1(x))


def main():
    x_train = torch.reshape(torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float), (-1, 2))
    y_train = torch.reshape(torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=float), (-1, 1))
    learning_rate = 0.01
    num_epoch = 100000

    model = Model()
    optimizer = torch.optim.SGD([model.w1, model.b1, model.w2, model.b2], lr=learning_rate)

    for epoch in range(num_epoch):
        # TODO replace forward with logit when figured out wht to do with logi
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model.logits(model.f1(x_train)), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(model.get_params())
    print(f"Loss: {loss}")

    # Plotting result
    title = "XOR function"
    fig = plt.figure(title)
    ax = fig.add_subplot(projection='3d')
    ax.set_title(title)
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
    ax.plot_wireframe(x1_grid, x2_grid, y_grid, color="red", label="$\\hat y = f(x1,x2) = sigmoid(x*W+b)$", rstride=5, cstride=5)
    ax.legend()

    # Showing plot

    plt.show()


if __name__ == '__main__':
    main()
