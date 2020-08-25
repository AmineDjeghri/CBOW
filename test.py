import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
import matplotlib.pyplot as plt
import sys

import models
from datasets import CBOWDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("runs/tensorboard")

def main():
    torch.manual_seed(1)
    CUDA = torch.cuda.is_available()
    if CUDA:
        print("avaible GPUs:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name())
    print("pytorch version: ", torch.__version__)

    dataset = CBOWDataset("data/en.txt",context_size=2,max_len=10)

    # Parameters
    CONTEXT_SIZE = 2
    EMBEDDING_SIZE = 300
    EPOCHS = 5
    LEARNING_RATE = 0.001

    # Model
    model = models.CBOW(len(dataset)+2*CONTEXT_SIZE, EMBEDDING_SIZE, CONTEXT_SIZE)
    if CUDA:
        model = model.cuda()

    # Training
    losses = train(model, dataset, EPOCHS, LEARNING_RATE)
    print(losses)


def train(model, dataset, epochs, lr):
    '''
    Train a model 

    Parameters:
        model (nn.module):
        data(list of tuples):  
        words_to_idx() :dict containing a mapping word->index
    Return:
        tensor (Tensor): 
    '''
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = []

    #### testing tensor board
    # data_loader=DataLoader(dataset,batch_size=2,shuffle=True)
    # examples=iter(data_loader)
    # example_context,example_target=examples.next()
    example_context,example_target=dataset[0]
    writer.add_graph(model, example_context)
    writer.close()

    for epoch in trange(epochs):
        total_loss = 0
        corrects=0
        for i in range(len(dataset)):
            context=dataset[i][0]
            target=dataset[i][1]
            output = model(context)
            loss = loss_func(output, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            corrects+= (predicted==target).sum().item()
        losses.append(total_loss)
        writer.add_scalar('training loss',total_loss/len(dataset),epoch*len(dataset)+i )
        writer.add_scalar('accuracy',corrects/len(dataset),epoch*len(dataset)+i )
    
    return losses


if __name__ == '__main__':
    main()
