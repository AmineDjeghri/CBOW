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

writer=SummaryWriter("runs/test2")

def main():
    torch.manual_seed(1)
    CUDA = torch.cuda.is_available()
    if CUDA:
        print("avaible GPUs:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name())
    print("pytorch version: ", torch.__version__)

    dataset = CBOWDataset("data/en.txt",context_size=2,max_len=15)

    # Parameters
    CONTEXT_SIZE = 2
    EMBEDDING_SIZE = 300
    EPOCHS = 5
    LEARNING_RATE = 0.01

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

    dataloader=DataLoader(dataset,batch_size=5,shuffle=True)
    example_context,example_target=next(iter(dataloader))
    print(example_context)
    writer.add_graph(model, example_context)
    writer.close()
    
    dataloader=DataLoader(dataset,batch_size=5,shuffle=True)

    for epoch in trange(epochs):
        total_loss = 0
        corrects=0
        for i,batch in enumerate(dataloader):
            print(i)
            context,target=batch
            output = model(context)
            loss = loss_func(output, target.view(-1))

            model.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            corrects+= (predicted==target).sum().item()
        losses.append(total_loss)
        writer.add_scalar('training loss',total_loss/len(dataloader),epoch*len(dataloader)+i )
        writer.add_scalar('accuracy',corrects/len(dataloader),epoch*len(dataloader)+i )
    
    return losses


if __name__ == '__main__':
    main()
