import torch
import torch.nn as nn

import unidecode
import string
import random
import re
import time, math

# Hyperparameters
num_epochs = 2000
print_every = 100
plot_every = 10
chunk_len = 200
hidden_size = 100
batch_size = 1
num_layers = 1
embedding_size = 70
lr = 0.002

# Prepare characters
all_characters = string.printable
n_characters = len(all_characters)
print(all_characters)
print('num_chars = ', n_characters)

# Get text data
file = unidecode.unidecode(open('filename').read())
file_len = len(file)
# print('file_len =', file_len)

# Random chunk
def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor

def random_training_set():
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

class RNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size

        self.encoder = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        out = self.encoder(input.view(batch_size, -1))
        out, (hidden, cell) = self.rnn(out, (hidden, cell)) # Add cell status
        out = self.decoder(out.view(batch_size, -1))
        return out, hidden, cell

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, batch_size, hidden_size)
        cell = torch.zeros(num_layers, batch_size, hidden_size) # Add cell status
        return hidden, cell

model = RNN(input_size=n_characters,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            output_size=n_characters,
            num_layers=2)

# Modeling
inp = char_tensor("A")
hidden,cell = model.init_hidden()
out,hidden,cell = model(inp,hidden,cell)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

# Train
for i in range(num_epochs):
    total = char_tensor(randum_chunk())
    inp = total[:-1]
    label = total[1:]
    hidden = model.init_hidden()

    loss = torch.tensor([0]).type(torch.FloatTensor)
    optimizer.zero_grad()
    for j in range(chunk_len-1):
        x = inp[j]
        y_ = label[j].unsqueeze(0).type(torch.LongTensor)
        y, hidden = model(x, hidden)
        loss += loss_func(y,y_)

    loss.backward()
    optimizer.step()

# Test Function
def test():
    start_str = "b"
    inp = char_tensor(start_str)
    hidden,cell = model.init_hidden()
    x = inp

    print(start_str,end="")
    for i in range(200):
        output,hidden,cell = model(x,hidden,cell)

        output_dist = output.data.view(-1).div(0.8).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = all_characters[top_i]

        print(predicted_char,end="")

        x = char_tensor(predicted_char)