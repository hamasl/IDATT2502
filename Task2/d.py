import torch
import torchvision
import matplotlib.pyplot as plt


class Model(torch.nn.Module):
    def __init__(self):
        # Using zeroes instead of rand for deterministic results
        self.num_of_digits = 10
        self.w = torch.zeros([784, self.num_of_digits], dtype=float, requires_grad=True)
        self.b = torch.zeros([1, self.num_of_digits], dtype=float, requires_grad=True)

    def get_params(self):
        return self.w, self.b

    def logits(self, x):
        return x @ self.w + self.b

    def forward(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.forward(x).argmax(1), y.argmax(1)).double())


def main():
    model = Model()

    # Load observations from the mnist dataset. The observations are divided into a training set and a test set
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
    x_train = mnist_train.data.reshape(-1, 784).double()  # Reshape input
    y_train = torch.zeros((mnist_train.targets.shape[0], model.num_of_digits))  # Create output tensor
    y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output


    num_epochs = 500
    learning_rate = 0.00001

    optimizer = torch.optim.SGD([model.w, model.b], lr=learning_rate)
    loss_fun = torch.nn.CrossEntropyLoss()
    for i in range(num_epochs):
        y_hat = model.logits(x_train)
        # y_train cannot be one hot encoded it needs to have labels.
        # Therefore .argmax(1) selects the index with the maximum value for each row, returning a 60000x1 matrix
        # loss takes y_hat that has to be logits
        loss = loss_fun(y_hat, y_train.argmax(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Test data loss {loss}")
    print(model.get_params())

    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
    x_test = mnist_test.data.reshape(-1, 784).double()  # Reshape input
    y_test = torch.zeros((mnist_test.targets.shape[0], model.num_of_digits), dtype=float)  # Create output tensor
    y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output
    print(f"Accuracy: {model.accuracy(x_test, y_test)}")

    plt.figure("Values of w")

    for i in range(model.num_of_digits):
        # Plotting 2 rows of five pictures:
        ax = plt.subplot(2, 5, i+1)
        ax.title.set_text(f"W = {i}")
        ax.imshow(model.w[:, i].reshape(28, 28).detach().numpy())
        plt.imsave(f"./img/w_{i}.png", model.w[:, i].reshape(28, 28).detach().numpy())
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


if __name__ == "__main__":
    main()
