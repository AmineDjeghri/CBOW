
import models
from datasets import CBOWDataset, WordsIdxDataset
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from datetime import datetime

def predict(text, model,nb_words_returned,dataset):
    """"
    Predict a word from a given context

    Parameters: 
        context(List of string): 
        model:
        nb_top: the number of words returned
        words_to_idx (dict):  map idx ->word
    Return:
        Top words(list of String): nb_top predicted words
    """
    context=text.replace('<mask>', '').lower().split()
    print(context)
    context_tensor=torch.tensor(dataset.get_idx_by_words(context)).view(1,-1)
    model.eval()
    prediction = model(context_tensor)
    indices = torch.sort(prediction, 1)[1].view(-1)
    indices=indices.tolist()
    top_indices=indices[:nb_words_returned]
    top_words=dataset.get_words_by_idx(top_indices)
    print(top_words)
    return top_words

experience="experience_2020_08_26_214808"
path_experience="./logs/"+experience+"/"

words_to_idx=utils.load_obj(path_experience+"words_to_idx.pkl")

words_idx_dataset=WordsIdxDataset(words_to_idx)

model= models.CBOW(len(words_to_idx), embedding_size=300, context_size=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

checkpoint=utils.load_checkpoint(path_experience+"/current_checkpoint.pt", model,optimizer)
model=checkpoint[0]
optimizer=checkpoint[1]

text=" the a and of  "
predict(text, model, nb_words_returned=10, dataset=words_idx_dataset)