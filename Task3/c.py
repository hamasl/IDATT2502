import torch
import torch.nn as nn
import run_model


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        self.logits = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, padding=2),
                                    nn.MaxPool2d(kernel_size=2),
                                    #nn.Dropout(p=0.5),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, kernel_size=5, padding=2),
                                    nn.MaxPool2d(kernel_size=2),
                                    #nn.Dropout(p=0.5),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(64 * 7 * 7, 1024),
                                    nn.Flatten(),
                                    nn.Linear(1024, 10))

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


if __name__ == "__main__":
    run_model.run_model(ConvolutionalNeuralNetworkModel())
