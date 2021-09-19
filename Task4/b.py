import numpy as np
import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, 7)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 4, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out)

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1, dtype=float)

    def loss(self, x,
             y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        # print(self.logits(x).shape)
        # print(y.argmax(1).shape)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


def get_emoji(input, model, index_to_emoji, index_to_char, char_encodings):
    model.reset()
    y = -1
    for i in range(len(input)):
        for j in range(len(index_to_char)):
            if index_to_char[j] == input[i]:
                y = model.f(torch.tensor(char_encodings[j])).argmax(1)
    return index_to_emoji[y]


if __name__ == "__main__":

    index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']
    encoding_size = len(index_to_char)
    char_encodings = np.eye(encoding_size)

    # U000.... is the baseball cap emoji
    index_to_emoji = ["\N{top hat}", "\N{rat}", "\N{cat}", "\N{office building}", "\N{man}", "\U0001F9E2", "\N{child}"]
    emoji_encodings = np.eye(len(index_to_emoji))

    x_train = torch.reshape(torch.tensor([[char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[0]],
                                          [char_encodings[4], char_encodings[2], char_encodings[3], char_encodings[0]],
                                          [char_encodings[5], char_encodings[2], char_encodings[3], char_encodings[0]],
                                          [char_encodings[6], char_encodings[7], char_encodings[2], char_encodings[3]],
                                          [char_encodings[8], char_encodings[2], char_encodings[3], char_encodings[3]],
                                          [char_encodings[5], char_encodings[2], char_encodings[9], char_encodings[0]],
                                          [char_encodings[10], char_encodings[11], char_encodings[12],
                                           char_encodings[0]]],
                                         dtype=torch.float), (7, 4, encoding_size))
    # Should maybe not be encoding size
    y_train = torch.reshape(torch.tensor([emoji_encodings], dtype=torch.float), (7, 1, len(index_to_emoji)))
    #print(y_train.shape)
    #print(y_train)

    model = LongShortTermMemoryModel(encoding_size)

    optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
    for epoch in range(500):
        model.reset()
        #print(x_train)
        #print(y_train)
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()
    print(get_emoji('rt', model, index_to_emoji, index_to_char, char_encodings))
    print(get_emoji('rats', model, index_to_emoji, index_to_char, char_encodings))