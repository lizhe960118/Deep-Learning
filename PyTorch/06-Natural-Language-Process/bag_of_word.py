#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/15 14:54
@Author  : LI Zhe
"""
# Skip-grams(SG)：预测上下文
    # 一个中心词一个上下文词迭代
    # 很多个不同的老师教一个学生， 德智体美全面发展
    # 更能捕捉低频词的特性和差异
# Continuous Bag of Words(CBOW): 预测目标单词
    # 一个中心词很多个上下文一次迭代
    # 一个老师教很多个不同学生，差异化较小

from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, n_dim) # Embedding矩阵表示m个词，每个词用d维表示
        self.project = nn.Linear(n_dim, n_dim, bias=False)
        self.linear1 = nn.Linear(n_dim, 128)
        self.linear2 = nn.Linear(128, n_word)

    def forward(self, x):
        x = self.embedding(x)
        x = self.project(x)
        x = torch.sum(x, 0, keepdim=True)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x = F.log_softmax(x) # 输出最有可能的单词
        return x

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
vocab = set(raw_text)
word_to_idx = {word:i for i, word in enumerate(vocab)}
idx_to_word = {word_to_idx[word]:word for word in word_to_idx}
data = []
CONTEXT_SIZE = 2
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context =[raw_text[i - 2], raw_text[i -1], raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

model = CBOW(len(word_to_idx), 100, CONTEXT_SIZE)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(100):
    print('epoch {}'.format(epoch))
    print('*' * 10)
    running_loss = 0
    for word in data:
        context, target = word
        context = Variable(torch.LongTensor([word_to_idx[i] for i in context]))
        target = Variable(torch.LongTensor([word_to_idx[target]]))
        if torch.cuda.is_available():
            context = context.cuda()
            target = target.cuda()
        # forward
        out = model(context)
        loss = criterion(out, target)
        running_loss += loss.data
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss:{:.6f}'.format(running_loss / len(data)))

with torch.no_grad():
    for test_idx in [5, 10, 15, 20, 25, 30]:
        context, target = data[test_idx]
        context = Variable(torch.LongTensor([word_to_idx[i] for i in context]))
        # predict
        out = model(context)
        _, predict_word_index = torch.max(out, 1)
        predict_word = idx_to_word[predict_word_index.data[0].item()]
        print('the real word is {}, and the predict word is {}'.format(target, predict_word))

