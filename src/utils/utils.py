import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """From: https://github.com/Bjarten/early-stopping-pytorch"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, epoch=None):
        if isinstance(model, list):
            self.is_list = True
        else:
            self.is_list = False

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.is_list:
                self.save_checkpoint(val_loss, model[0], suffix="_T")
                self.save_checkpoint(val_loss, model[1], suffix="_C")
            else:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            if self.is_list:
                self.save_checkpoint(val_loss, model[0], suffix="_T")
                self.save_checkpoint(val_loss, model[1], suffix="_C")
            else:
                self.save_checkpoint(val_loss, model)
            self.counter = 0
        #print("best score: {}, counter: {}".format(self.best_score, self.counter))

    def save_checkpoint(self, val_loss, model, suffix=""):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if suffix == "":
            torch.save(model.state_dict(), self.path, _use_new_zipfile_serialization=False)
        else:
            torch.save(model.state_dict(), self.path+suffix, _use_new_zipfile_serialization=False)
        self.val_loss_min = val_loss

    #def get_best_weights(self, model):
    #    return model.load_state_dict(torch.load(self.path))