import torch
import torch.nn as nn
import math

# input = torch.FloatTensor([-0.4564,-0.53223, -0.63432])
input = torch.FloatTensor([-0.4089, -1.2471, 0.5907])
target = torch.FloatTensor([1, 0.3, 0.7])
target2 = torch.FloatTensor([0, 1, 1])
loss = nn.BCEWithLogitsLoss()
print(loss(input, target))
print(loss(input, target2))

input = [-0.4089, -1.2471, 0.5907]
#sum_input = sum([math.exp(x) for x in input])
#softmax_input = [math.exp(x) / sum_input for x in input]
#print(softmax_input)
sigmoid_input = [ 1 / (1 + math.exp(-x)) for x in input ]
target = [1, 0.3, 0.7]
def count_log(input, target):
    loss = 0
    for i in range(len(input)):
            loss += - (target[i] * (math.log(input[i]))) - ((1 - target[i]) * (math.log(1 - input[i])))
    return (loss / len(input))

print(count_log(sigmoid_input, target))
target2 = [0,1,1]
print(count_log(sigmoid_input, target2))
