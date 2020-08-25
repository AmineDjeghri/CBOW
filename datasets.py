from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CBOWDataset(Dataset):
    def __init__(self, file,context_size,max_len=None):
        self.file=file
        self.context_size=context_size
        self.max_len=max_len
        self._init_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        context,target = self.data[i]
        context_idx=self.get_idx_by_words(context)
        target_idx=self.get_idx_by_words([target])

        return context_idx,target_idx

    def _init_dataset(self):
        with open(self.file, "r", encoding="utf8") as f:
            print('file: ', f)
            text = f.read()

            self.data, self.words_to_idx = preprocess_text(text, context_size=2, max_len=self.max_len)
            self.idx_to_words = {v: k for k, v in self.words_to_idx.items()}
    
    def get_idx_by_words(self,words):
        '''
        Retrieve the indexes of given words

        Parameters:
            words (list of string): 
        
        Return:
            tensor (Tensor): tensor of indexes
        '''
        tensor = torch.LongTensor([self.words_to_idx[word] for word in words])
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        return tensor

    def get_words_by_idx(self,idx):
        """
        return the word of a given indices
        Parameters:
            idx (list of indices): 
        """
        words=[]
        for i in idx:
            words.append(self.idx_to_words[i])
        return words


def preprocess_text(text, context_size,max_len=None):
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

    if max_len:
        text=text[:max_len]
    
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


if __name__ == '__main__':
    dataset = CBOWDataset("data/en.txt",context_size=2,max_len=10)
    print(len(dataset))
    print(dataset[1])
    print(dataset.words_to_idx)
    print(dataset.get_idx_by_words(['upon','stories']))
    print(dataset.get_words_by_idx([6]))
    print(dataset.data)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=2)
    for i, batch in enumerate(dataloader):
        print(i, batch)