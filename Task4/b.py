import numpy as np
import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, word_len):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size*word_len, 128)  # 128 is the state size
        self.dense = nn.Linear(128, 7)  # 128 is the state size

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


def get_emoji(input, model, index_to_emoji, index_to_char, char_encodings):
    model.reset()
    tens = []
    for i in range(len(input)):
        for j in range(len(index_to_char)):
            if index_to_char[j] == input[i]:
                tens.append(char_encodings[j])
    while len(tens) < 4:
        tens.append(char_encodings[0])
    y = model.f(torch.tensor(tens, dtype=torch.float).reshape(1, 1, -1)).argmax(1)
    return index_to_emoji[y]



if __name__ == "__main__":

    index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']
    encoding_size = len(index_to_char)
    word_len = 4
    char_encodings = np.eye(encoding_size)

    # U000.... is the baseball cap emoji
    index_to_emoji = ["\N{top hat}", "\N{rat}", "\N{cat}", "\N{office building}", "\N{man}", "\U0001F9E2", "\N{child}"]
    emoji_encoding_size = len(index_to_emoji)
    emoji_encodings = np.eye(emoji_encoding_size)

    x_train = torch.reshape(torch.tensor([[char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[0]],
                                          [char_encodings[4], char_encodings[2], char_encodings[3], char_encodings[0]],
                                          [char_encodings[5], char_encodings[2], char_encodings[3], char_encodings[0]],
                                          [char_encodings[6], char_encodings[7], char_encodings[2], char_encodings[3]],
                                          [char_encodings[8], char_encodings[2], char_encodings[3], char_encodings[3]],
                                          [char_encodings[5], char_encodings[2], char_encodings[9], char_encodings[0]],
                                          [char_encodings[10], char_encodings[11], char_encodings[12],
                                           char_encodings[0]]
                                          ],
                                         dtype=torch.float), (emoji_encoding_size, word_len, encoding_size))

    y_train = torch.reshape(
        torch.tensor([emoji_encodings], dtype=torch.float),
        (emoji_encoding_size, 1, emoji_encoding_size))
    model = LongShortTermMemoryModel(encoding_size, word_len)

    optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
    for epoch in range(500):
        for emoji_data_index in range(emoji_encoding_size):
            model.reset()
            model.loss(x_train[emoji_data_index].reshape(1, 1, -1), y_train[emoji_data_index].reshape(1, -1)).backward()
            optimizer.step()
            optimizer.zero_grad()
    print(get_emoji('rt', model, index_to_emoji, index_to_char, char_encodings))
    print(get_emoji('rat', model, index_to_emoji, index_to_char, char_encodings))
    print(get_emoji('rats', model, index_to_emoji, index_to_char, char_encodings))
    print(get_emoji('hat', model, index_to_emoji, index_to_char, char_encodings))
    print(get_emoji('matt', model, index_to_emoji, index_to_char, char_encodings))
    print(get_emoji('flat', model, index_to_emoji, index_to_char, char_encodings))
    print(get_emoji('son', model, index_to_emoji, index_to_char, char_encodings))
    print(get_emoji('cap', model, index_to_emoji, index_to_char, char_encodings))
    print(get_emoji('cat', model, index_to_emoji, index_to_char, char_encodings))
    print(get_emoji('ct', model, index_to_emoji, index_to_char, char_encodings))
