#####
# Ross Dahlke
# Identifying Opinion Change in Deliberative Settings
# Stanford University, CS 230: Deep Learning
#####

# This script contains the code for the milestone
# Specifically, this document is for my baseline LSTM model

# https://medium.com/analytics-vidhya/spam-ham-classification-using-lstm-in-pytorch-950daec94a7c
import docx2txt
import pandas as pd
import numpy as np
import os
from collections import Counter

# transcripts
deltas = pd.read_csv("data/processed/survey/opinion_deltas.csv")
doc_list = []
delta_list = []
for id in deltas["id"][1:10]:
    try:
        doc = docx2txt.process("data/processed/transcripts/" + id + ".docx").replace("\n", " ").lower()
        doc_list.append(doc)
        delta = abs(deltas[deltas["id"] == id]["delta"].values[0])
        delta_list.append(delta)
    except Exception:
        print("passed on " + id + ". No transcript.")
        pass

vocabs = [vocab for seq in doc_list for vocab in seq.split()]
vocab_count = Counter(vocabs)
vocab_count = vocab_count.most_common(len(vocab_count))

vocab_to_int = {word : index+2 for index, (word, count) in enumerate(vocab_count)}
vocab_to_int.update({'__PADDING__': 0}) # index 0 for padding
vocab_to_int.update({'__UNKNOWN__': 1}) # index 1 for unknown word such as broken character

import torch
from torch.autograd import Variable

# Tokenize & Vectorize sequences
vectorized_seqs = []
for seq in doc_list:
  vectorized_seqs.append([vocab_to_int.get(word,1) for word in seq.split()])

# Save the lengths of sequences
seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))

# Add padding(0)
seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
  seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

print(seq_lengths.max()) # tensor(30772)
print(seq_tensor[0]) # tensor([ 20,  77, 666,  ...,   0,   0,   0])
print(seq_lengths[0]) # tensor(412)

import torch.utils.data.sampler as splr

class CustomDataLoader(object):
  def __init__(self, seq_tensor, seq_lengths, label_tensor, batch_size):
    self.batch_size = batch_size
    self.seq_tensor = seq_tensor
    self.seq_lengths = seq_lengths
    self.label_tensor = label_tensor
    self.sampler = splr.BatchSampler(splr.RandomSampler(self.label_tensor), self.batch_size, False)
    self.sampler_iter = iter(self.sampler)

  def __iter__(self):
    self.sampler_iter = iter(self.sampler) # reset sampler iterator
    return self

  def _next_index(self):
    return next(self.sampler_iter) # may raise StopIteration

  def __next__(self):
    index = self._next_index()

    subset_seq_tensor = self.seq_tensor[index]
    subset_seq_lengths = self.seq_lengths[index]
    subset_label_tensor = self.label_tensor[index]

    subset_seq_lengths, perm_idx = subset_seq_lengths.sort(0, descending=True)
    subset_seq_tensor = subset_seq_tensor[perm_idx]
    subset_label_tensor = subset_label_tensor[perm_idx]

    return subset_seq_tensor, subset_seq_lengths, subset_label_tensor

  def __len__(self):
    return len(self.sampler)

# shuffle data
delta_array = np.asarray(delta_list)
shuffled_idx = torch.randperm(delta_array.shape[0])

seq_tensor = seq_tensor[shuffled_idx]
seq_lenghts = seq_lengths[shuffled_idx]
label = delta_array[shuffled_idx]

# divide data into 3 sets
PCT_TRAIN = 0.7 # 70% of data will be train set
PCT_VALID = 0.2 # 20% of data will be validation set
# The rest of data will be test set

length = len(label)
train_seq_tensor = seq_tensor[:int(length*PCT_TRAIN)]
train_seq_lengths = seq_lengths[:int(length*PCT_TRAIN)]
train_label = label[:int(length*PCT_TRAIN)]

valid_seq_tensor = seq_tensor[int(length*PCT_TRAIN):int(length*(PCT_TRAIN+PCT_VALID))]
valid_seq_lengths = seq_lengths[int(length*PCT_TRAIN):int(length*(PCT_TRAIN+PCT_VALID))]
valid_label = label[int(length*PCT_TRAIN):int(length*(PCT_TRAIN+PCT_VALID))]

test_seq_tensor = seq_tensor[int(length*(PCT_TRAIN+PCT_VALID)):]
test_seq_lengths = seq_lengths[int(length*(PCT_TRAIN+PCT_VALID)):]
test_label = label[int(length*(PCT_TRAIN+PCT_VALID)):]

print(train_seq_tensor.shape) # torch.Size([4200, 30772])
print(valid_seq_tensor.shape) # torch.Size([1199, 30772])
print(test_seq_tensor.shape) # torch.Size([601, 30772])

# Instantiate data loaders
batch_size = 70
train_loader = CustomDataLoader(train_seq_tensor, train_seq_lengths, train_label, batch_size)
valid_loader = CustomDataLoader(valid_seq_tensor, valid_seq_lengths, valid_label, batch_size)
test_loader = CustomDataLoader(test_seq_tensor, test_seq_lengths, test_label, batch_size)

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SpamHamLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, n_layers,\
                 drop_lstm=0.1, drop_out = 0.1):

        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_lstm, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(drop_out)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()


    def forward(self, x, seq_lengths):

        # embeddings
        embedded_seq_tensor = self.embedding(x)

        # pack, remove pads
        packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)

        # lstm
        packed_output, (ht, ct) = self.lstm(packed_input, None)
          # https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html
          # If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero

        # unpack, recover padded sequence
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # collect the last output in each batch
        last_idxs = (input_sizes - 1).to(device) # last_idxs = input_sizes - torch.ones_like(input_sizes)
        output = torch.gather(output, 1, last_idxs.view(-1, 1).unsqueeze(2).repeat(1, 1, self.hidden_dim)).squeeze() # [batch_size, hidden_dim]

        # dropout and fully-connected layer
        output = self.dropout(output)
        output = self.fc(output).squeeze()

        # sigmoid function
        output = self.sig(output)

        return output

seqs = torch.tensor([[1,2,3,4,5], [6,7,8,0,0]])
lengths = torch.tensor([5,3], dtype = torch.int64).cpu()  # should be a 1D / CPU / int64 tensor
result = pack_padded_sequence(seqs, lengths, batch_first=True)
print(result.data) # tensor([1, 6, 2, 7, 3, 8, 4, 5])
print(result.batch_sizes) # tensor([2, 2, 2, 1, 1])

# seq_1) 1 2 3 4 5
# seq_2) 6 7 8 0 0
# batch) 2 2 2 1 1

# training params


vocab_size = len(vocab_to_int)
embedding_dim = 100 # int(vocab_size ** 0.25) # 15
hidden_dim = 15
output_size = 1
n_layers = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
net = SpamHamLSTM(vocab_size, embedding_dim, hidden_dim, output_size, n_layers, \
                 0.2, 0.2)
net = net.to(device)
print(net)

# loss and optimization functions
criterion = nn.BCELoss()

lr=0.03
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,\
                                                       mode = 'min', \
                                                      factor = 0.5,\
                                                      patience = 2)

epochs = 6

counter = 0
print_every = 1
clip=5 # gradient clipping


net.train()
# train for some number of epochs
val_losses = []
for e in range(epochs):

    scheduler.step(e)

    for seq_tensor, seq_tensor_lengths, label in iter(train_loader):
        counter += 1

        seq_tensor = seq_tensor.to(device)
        seq_tensor_lengths = seq_tensor_lengths.to(device)
        label = torch.tensor(label).to(device)

        # get the output from the model
        output = net(seq_tensor, seq_tensor_lengths)

        # get the loss and backprop
        loss = criterion(output, label.float())
        optimizer.zero_grad()
        loss.backward()

        # prevent the exploding gradient
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss

            val_losses_in_itr = []
            sums = []
            sizes = []

            net.eval()

            for seq_tensor, seq_tensor_lengths, label in iter(valid_loader):

                seq_tensor = seq_tensor.to(device)
                seq_tensor_lengths = seq_tensor_lengths.to(device)
                label = torch.tensor(label).to(device)
                output = net(seq_tensor, seq_tensor_lengths)

                # losses
                val_loss = criterion(output, label.float())
                val_losses_in_itr.append(val_loss.item())

                # accuracy
                binary_output = (output >= 0.5).short() # short(): torch.int16
                right_or_not = torch.eq(binary_output, label)
                sums.append(torch.sum(right_or_not).float().item())
                sizes.append(right_or_not.shape[0])

            accuracy = sum(sums) / sum(sizes)

            net.train()
            print("Epoch: {:2d}/{:2d}\t".format(e+1, epochs),
                  "Steps: {:3d}\t".format(counter),
                  "Loss: {:.6f}\t".format(loss.item()),
                  "Val Loss: {:.6f}\t".format(np.mean(val_losses_in_itr)),
                  "Accuracy: {:.3f}".format(accuracy))
