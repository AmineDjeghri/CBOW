import torch
import shutil
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
import pickle


def save_checkpoint(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_checkpoint(checkpoint_fpath, model, optimizer):
    map_location=torch.device('cpu')
    if torch.cuda.is_available():
        map_location=None
    checkpoint = torch.load(checkpoint_fpath, map_location)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min


def save_obj( name, obj):
    with open( name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)


def plot_confusion_matrix(consfusion_matrix,labels):
    print(consfusion_matrix)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(consfusion_matrix, vmax=1)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    # ax.set_xticklabels([''] + labels)
    # ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.gca().set_aspect('auto')
    plt.show()