import torch
import torch.nn.functional
import matplotlib
import matplotlib.pyplot as plt


# Henter inn data
# Lager en modell med valgt w og b
# Tester loss for modellen
# Optimaliserer w og b

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.b = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))

    def forward(self, x):
        return self.w @ x + self.b


def main():
    x_train = torch.reshape(torch.tensor([1, 2, 2], dtype=torch.float), (1, 3))
    y_train = torch.reshape(torch.tensor([[1,2,2]], dtype=torch.float), (1,3))
    learning_rate = 0.1
    num_epoch = 10

    model = Model()
    loss_fun = torch.nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD([model.w, model.b], lr=learning_rate)

    for epoch in range(num_epoch):
        y_hat = model(x_train)
        loss = loss_fun(y_train, y_hat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(model.state_dict())
    print(loss_fun(y_train, y_hat))

    plt.title("Linear regression 2d")
    plt.show()


if __name__ == '__main__':
    main()
