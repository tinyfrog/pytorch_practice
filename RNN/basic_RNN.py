import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

n_hidden = 35
lr = 0.01
epochs = 1000

string = "hello pytorch. how long can a rnn cell remember?"
chars = "abcdefghijklmnopqrstuvwxyz ?!.,:;01"
char_list = [i for i in chars] # useful
n_letters = len(char_list)

'''
convert into one-hot vector
Start = [0 0 0 ... 1 0]
a = [1 0 0 ... 0 0]
b = [0 1 0 ... 0 0]
c = [0 0 1 ... 0 0]
end = [0 0 0 ... 0 1]
'''

def string_to_onehot(string):
    start = np.zeros(shape=len(char_list), dtype=int)
    end = np.zeros(shape=len(char_list), dtype=int)
    start[-2] = 1
    end[-1] = 1
    for i in string:
        idx = char_list.index(i)
        zero = np.zeros(shape=n_letters, dtype=int)
        zero[idx] = 1
        start = np.vstack([start, zero])
    output = np.vstack([start, end])
    '''
    np.vstack 
    Connect two or more arrays with the same number of columns 'up and down' to create an array with more rows.
    '''
    return output

def onehot_to_word(onehot_1):
    onehot = torch.Tensor.numpy(onehot_1)
    return char_list[onehot.argmax()]

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.act_fn = nn.Tanh()

    def forward(self, input, hidden):
        hidden = self.act_fn(self.i2h(input) + self.h2h(hidden))
        output = self.i2o(hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = RNN(n_letters, n_hidden, n_letters)

loss_func = nn.MSELoss() # L2 Loss function
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

one_hot = torch.from_numpy(string_to_onehot(string)).type_as(torch.FloatTensor())

for i in range(epochs):
    rnn.zero_grad()
    total_loss = 0
    hidden = rnn.init_hidden()

    for j in range(one_hot.size()[0]-1):
        input_ = one_hot[j:j+1,:]
        target = one_hot[j+1]

        output, hidden = rnn.forward(input_, hidden)
        loss = loss_func(output.view(-1), target.view(-1))
        total_loss += loss
        input_ = output

    total_loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(total_loss)

start = torch.zeros(1, len(char_list))
start[:,-2] = 1

with torch.no_grad():
    hidden = rnn.init_hidden()
    input_ = start
    output_string = ""
    for i in range(len(string)):
        output, hidden = rnn.forward(input_, hidden)
        output_string += onehot_to_word(output.data)
        input_ = output

print(output_string)

