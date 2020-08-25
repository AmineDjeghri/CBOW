import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CBOW(nn.Module):

    """"
    Word2Vec CBOW model
    """

    def __init__(self, vocab_size, embedding_size, context_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        if torch.cuda.is_available():
            self.embeddings = self.embeddings.cuda()
        self.net = nn.Sequential(
            nn.Linear(context_size*embedding_size*2, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size))
        
    def forward(self, inputs):
        batch=inputs.shape[0]
        embedded = self.embeddings(inputs)
        embedded = embedded.view((batch, -1))
        out = self.net(embedded)
        return out
