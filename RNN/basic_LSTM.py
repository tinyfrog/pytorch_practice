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
