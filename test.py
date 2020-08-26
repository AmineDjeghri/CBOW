import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import shutil
from tqdm import trange
import matplotlib.pyplot as plt
import sys
from tensorboardX import SummaryWriter
import models
from datasets import CBOWDataset,train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import utils

def main():
    torch.manual_seed(1)
    CUDA = torch.cuda.is_available()
    if CUDA:
        print("avaible GPUs:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name())
    print("pytorch version: ", torch.__version__)


    # Parameters
    CONTEXT_SIZE = 2
    EMBEDDING_SIZE = 300
    EPOCHS = 10
    LEARNING_RATE = 0.001

    #dataset
    dataset = CBOWDataset("data/en.txt",context_size=2,max_len=100)
    print("dataset len: ",len(dataset))

    # Model
    model = models.CBOW(len(dataset)+2*CONTEXT_SIZE, EMBEDDING_SIZE, CONTEXT_SIZE)
    if CUDA:
        model = model.cuda()

    # Training
    train(model, dataset, EPOCHS, LEARNING_RATE)



def train(model, dataset, epochs, lr):
    checkpoint_path="./logs/current_checkpoint.pt"
    best_model_path="./logs/best_model.pt"
    valid_loss_min=np.Inf
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    dataloader=DataLoader(dataset,batch_size=1,shuffle=True)
    print(len(dataloader))
    example_context,example_target=next(iter(dataloader))

    writer=SummaryWriter("runs/100k")
    #writer.add_graph(model, example_context)

    train_loader,valid_loader = train_test_split(dataset, batch_size=4, validation_split=0.2, shuffle=True)

    for epoch in trange(epochs):
        train_loss = 0
        valid_loss = 0

        # train
        model.train()
        for i,batch in enumerate(train_loader):
            context,target=batch
            target=target.view(-1)
            output = model(context)
    
            loss = criterion(output, target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            corrects= (predicted==target).sum().item()
            writer.add_scalar('training loss',loss.item()/len(batch[0]),epoch*len(dataloader)+i )
            writer.add_scalar('accuracy',corrects/len(batch[0]),epoch*len(dataloader)+i )
        
        # validate
        model.eval()
        for i,batch in enumerate(valid_loader):
            context,target=batch
            target=target.view(-1)
            output = model(context)
    
            loss = criterion(output, target)

            valid_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            corrects= (predicted==target).sum().item()
            writer.add_scalar('validating loss',loss.item()/len(batch[0]),epoch*len(dataloader)+i )
            writer.add_scalar('validating accuracy',corrects/len(batch[0]),epoch*len(dataloader)+i )

        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)

        print('Epoch: {} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))

        # create checkpoint 
        checkpoint = {
            'next_epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # save checkpoint
        utils.save_checkpoint(checkpoint, False, checkpoint_path, best_model_path)
        
        ## save the model as the best model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                
            utils.save_checkpoint(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss
            
    writer.close()
    return model
    

if __name__ == '__main__':
    main()
