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
from datetime import datetime
import os

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
    dt=datetime.now().strftime('%Y_%m_%d_%H%M%S')+"/"
    path_experience="./logs/experience_"+dt
    path_runs="runs/experience_"+dt
    os.makedirs(path_experience, exist_ok=True)
    
    #dataset
    dataset = CBOWDataset("data/big.txt",context_size=2,max_len=1000)
    utils.save_obj(path_experience+"words_to_idx.pkl",dataset.words_to_idx)
    print("unique vocab: ",dataset.len_vocab)

    # Model
    model = models.CBOW(dataset.len_vocab, EMBEDDING_SIZE, CONTEXT_SIZE)
    if CUDA:
        model = model.cuda()

    # Training
    train(model, dataset, EPOCHS, LEARNING_RATE,path_experience,path_runs)



def train(model, dataset, epochs, lr,path_experience,path_runs):
    checkpoint_path=path_experience+"/current_checkpoint.pt"
    best_model_path=path_experience+"/best_model.pt"
    valid_loss_min=np.Inf
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    labels=list(dataset.words_to_idx.keys())

    dataloader=DataLoader(dataset,batch_size=1,shuffle=True)
    print('number of total words in ',len(dataloader))
    example_context,example_target=next(iter(dataloader))

    writer=SummaryWriter(path_runs)
    #writer.add_graph(model, example_context)

    train_loader,valid_loader = train_test_split(dataset, batch_size=10, validation_split=0.2, shuffle=True)

    for epoch in trange(epochs):
        train_loss = 0
        valid_loss = 0
        nb_classes=dataset.len_vocab
        confusion_matrix=torch.zeros(nb_classes,nb_classes)
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
            writer.add_scalar('training loss',loss.item()/len(batch[0]),epoch*len(train_loader)+i )
            writer.add_scalar('accuracy',corrects/len(batch[0]),epoch*len(train_loader)+i )

        # validate
        model.eval()
        class_probs = []
        class_preds = []
        for i,batch in enumerate(valid_loader):
            context,target=batch
            target=target.view(-1)
            output = model(context)
    
            loss = criterion(output, target)

            valid_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            corrects= (predicted==target).sum().item()
            writer.add_scalar('validating loss',loss.item()/len(batch[0]),epoch*len(valid_loader)+i )
            writer.add_scalar('validating accuracy',corrects/len(batch[0]),epoch*len(valid_loader)+i )
            for t,p in zip(target,predicted):
                confusion_matrix[t,p]+=1
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            class_probs.append(class_probs_batch)
            class_preds.append(predicted)

        #pr curve
        with torch.no_grad():
            test_probs = torch.cat([torch.stack(b) for b in class_probs])
            test_preds = torch.cat(class_preds)
            for i in range(dataset.len_vocab):
                tensorboard_preds = test_preds == i
                tensorboard_probs = test_probs[:, i]

                writer.add_pr_curve(tag=dataset.get_words_by_idx([i])[0],
                                    labels=tensorboard_preds,
                                    predictions=tensorboard_probs,
                                    global_step=epoch)
        
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        print('Epoch: {} \t Training Loss: {:.6f} \t Validation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))

        # create checkpoint 
        checkpoint = {
            'epoch': epoch,
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
        
    utils.plot_confusion_matrix(confusion_matrix,labels)
    writer.close()
    return model

if __name__ == '__main__':
    main()
