import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def preprocess_text(text, context_size):
    '''
    Convert text to data:(context, target) for training cbow model

    Parameters:
        text (String): text to preprocess
        context_size (int): the context window 
    
    Return:
        data (tuple): data in form of (context, target)
        words_to_idx(dict): dict containing a mapping word->index
    '''
    text = text.lower().split()

    # Build contexts and targets
    data = list() 
    for i in range(context_size, len(text) - context_size):
        context = text[i-context_size: i] + text[i+1: i+context_size+1]
        target = text[i]  
        data.append((context, target))
    
    # Map words to index
    vocab = set(text)
    words_to_idx = {w: i for i, w in enumerate(vocab)}

    return data, words_to_idx


def get_idx_by_word(words, word_to_idx):
    '''
    Retrieve the indexes of given words

    Parameters:
        words (list of string): 
    
    Return:
        tensor (Tensor): tensor of indexes
    '''
    tensor = torch.LongTensor([word_to_idx[word] for word in words])
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor
