import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
import matplotlib.pyplot as plt

import models
import utils


def main():
    torch.manual_seed(1)
    CUDA = torch.cuda.is_available()
    if CUDA:
        print("avaible GPUs:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name())
    print("pytorch version: ", torch.__version__)

    with open("data/en.txt", "r", encoding="utf8") as f:
        print('file: ', f)
        text = f.read()

    # Preprocessing
    data, words_to_idx = utils.preprocess_text(text, context_size=2)
    idx_to_words = {v: k for k, v in words_to_idx.items()}

    # Parameters
    CONTEXT_SIZE = 2
    EMBEDDING_SIZE = 300
    EPOCHS = 5
    LEARNING_RATE = 0.001

    # Model
    model = models.CBOW(len(words_to_idx),EMBEDDING_SIZE,CONTEXT_SIZE)
    if CUDA:
        model = model.cuda()

    # Training
    losses = train(model, data,words_to_idx, EPOCHS,LEARNING_RATE)
    print(losses)


def train(model, data, words_to_idx, epochs, lr):
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
    losses=[]

    for epoch in trange(epochs):
        total_loss = 0
        for context, target in data:
            context_idx = utils.get_idx_by_word(context, words_to_idx)
            target_idx = utils.get_idx_by_word([target], words_to_idx)

            output = model(context_idx)
            loss = loss_func(output, target_idx)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)
    
    return losses


if __name__ == '__main__':
    main()
