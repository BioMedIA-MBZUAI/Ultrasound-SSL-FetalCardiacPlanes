import sys, os
import random
import numpy as np
import torch


def START_SEED(seed=73):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_pretrained(model, weight_path, flexible = False):
    if not weight_path:
        print("No weight file to be loaded returning Model with Random weights")
        return model

    model_dict = model.state_dict()
    weight_dict = torch.load(weight_path)

    if weight_dict.has_key('model'):
        pretrain_dict = weight_dict['model']
    else:
        pretrain_dict = weight_dict

    if flexible:
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    if not len(pretrain_dict):
        raise Exception(f"No weight names match to be loaded; though file exits ! {weight_path}, Dict: {weight_dict}")

    print("Pretrained layers:", pretrain_dict.keys())
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    return model

def count_train_param(model):
    train_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(train_params_count))
    return train_params_count

def freeze_params(model, exclusion_list = []):
    ## TODO: Exclusion lists
    for param in model.parameters():
        param.requires_grad = False
    return model




class ObjDict(dict):
    """
    reference: https://stackoverflow.com/a/32107024
    """
    def __init__(self, *args, **kwargs):
        super(ObjDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(ObjDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(ObjDict, self).__delitem__(key)
        del self.__dict__[key]