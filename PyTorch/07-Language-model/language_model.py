#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/16 10:34
@Author  : LI Zhe
"""
import torch
from torch import nn, optim
from torch.autograd import Variable
from data_utils import Corpus

class language_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super(language_model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embedding(x)
        x, hi = self.lstm(x, h)
        b, s, h = x.size()
        x = x.contiguous().view(b * s, h)
        x = self.linear(x)
        return x, hi

seq_length = 30

train_file = 'train.txt'
val_file = 'val.txt'
test_file = 'test.txt'
train_corpus = Corpus()
val_corpus = Corpus()
test_corpus = Corpus()

train_id = train_corpus.get_data(train_file)
val_id = train_corpus.get_data(val_file)
test_id = train_corpus.get_data(test_file)

vocab_size = len(train_corpus.dic)
num_batches = train_id.size(1) // seq_length

model = language_model(vocab_size, 128, 1024, 1)

# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def detach(states):
    if torch.cuda.is_available():
        return [Variable(state.data).cuda() for state in states]
    else:
        return [Variable(state.data) for state in states]

for epoch in range(5):
    print('*' * 10)
    print('epoch {}'.format(epoch))
    running_loss = 0
    if torch.cuda.is_available():
        states = (Variable(torch.zeros(1, 20, 1024)).cuda(),
                  Variable(torch.zeros(1, 20, 1024)).cuda())
    else:
        states = (Variable(torch.zeros(1, 20, 1024)),
                  Variable(torch.zeros(1, 20, 1024)))
    for i in range(0, train_id.size(1) - 2 * seq_length, seq_length):
        input_x = train_id[:, i : (i + seq_length)]
        label = train_id[:, (i + seq_length):(i + 2 * seq_length)]
        if torch.cuda.is_available():
            input_x = Variable(input_x).cuda()
            label = Variable(label).cuda()
        else:
            input_x = Variable(input_x)
            label = Variable(label)
        # print(label.size(0), label.size(1))
        label = label.contiguous().view(label.size(0) * label.size(1), -1)
        states = detach(states)
        # forward
        out, states = model(input_x, states)
        loss = criterion(out, label.view(-1))
        running_loss += loss.data
        # backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        step = (i + 1) // seq_length
        if step % 100 == 0:
            print('epoch [{} / {}], step[{} / {}], loss: {}'.format(epoch+1, 5, step, num_batches, loss.data))
    print('Epoch {} Finished, loss:{}'.format(epoch+1, running_loss/(train_id.size(1) // seq_length - 1)))

# Test model
model.eval()
# eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
eval_loss = 0.0
eval_acc = 0.0

with torch.no_grad():
    if torch.cuda.is_available():
        states = (Variable(torch.zeros(1, 20, 1024)).cuda(),
                  Variable(torch.zeros(1, 20, 1024)).cuda())
    else:
        states = (Variable(torch.zeros(1, 20, 1024)),
                  Variable(torch.zeros(1, 20, 1024)))

    for i in range(0, test_id.size(1) - 2 * seq_length, seq_length):
        input_x = test_id[:, i: (i + seq_length)]
        label = test_id[:, (i + seq_length):(i + 2 * seq_length)]
        if torch.cuda.is_available():
            input_x = Variable(input_x).cuda()
            label = Variable(label).cuda()
        else:
            input_x = Variable(input_x)
            label = Variable(label)
        label = label.contiguous().view(label.size(0) * label.size(1), -1)
        states = detach(states)

        # forward
        out, states = model(input_x, states)
        test_loss = criterion(out, label.view(-1))

        step = (i + 1) // seq_length
        if step % 100 == 0:
            print('step[{}], loss: {}'.format(step, test_loss.data))
    print('Test Loss: {:.6f}'.format(eval_loss / (test_id.size(1) // seq_length - 1)))
