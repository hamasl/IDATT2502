import numpy as np
import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1, dtype=float)

    def loss(self, x,
             y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)

        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


def test_model(model):
    # Generate characters from the initial characters ' h'
    model.reset()
    text = ' h'
    model.f(torch.tensor([[char_encodings[0]]], dtype=torch.float))
    y = model.f(torch.tensor([[char_encodings[1]]], dtype=torch.float))
    text += index_to_char[y.argmax(1)]
    for c in range(50):
        y = model.f(torch.tensor([[char_encodings[y.argmax(1)]]], dtype=torch.float))
        text += index_to_char[y.argmax(1)]
    print(text)


if __name__ == "__main__":
    char_encodings = np.eye(8)
    encoding_size = len(char_encodings)

    index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']

    x_train = torch.reshape(torch.tensor([char_encodings[0], char_encodings[1], char_encodings[2], char_encodings[3],
                                          char_encodings[3], char_encodings[4], char_encodings[0], char_encodings[5],
                                          char_encodings[4],
                                          char_encodings[6], char_encodings[3], char_encodings[7]], dtype=torch.float),
                            (12, 1, 8))  # ' hello world'

    y_train = torch.tensor(
        [char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3], char_encodings[4],
         char_encodings[0], char_encodings[5], char_encodings[4], char_encodings[6], char_encodings[3],
         char_encodings[7], char_encodings[0]], dtype=torch.float)  # 'hello world '

    model = LongShortTermMemoryModel(encoding_size)

    optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
    for epoch in range(500):
        model.reset()
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 10 == 9:
            test_model(model)
    print("Final result:")
    test_model(model)
