""" Barlow Twin self-supervision training
"""
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from tqdm import tqdm

from torch import nn, optim
import torch
import torchvision
import torchsummary

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl
from algorithms.lars_optim import LARS, adjust_learning_rate
from algorithms.barlowtwins import BarlowTwins
from datacode.natural_image_data import Cifar100Dataset
from datacode.augmentations import BarlowTwinsTransformOrig


print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device Used:", device)

###============================= Configure and Setup ===========================

CFG = rutl.ObjDict(
use_amp = True, #automatic Mixed precision

datapath= "/home/USR/WERK/data/",
epochs= 1000,
batch_size= 2048,

learning_rate_weights = 0.2,
learning_rate_biases  = 0.0048,
weight_decay = 1e-6,
lmbd = 0.0051,
image_size=256,

featx_arch = "resnet50", # "resnet34/50/101"
featx_pretrain =  None, # path-to-weights or None
projector = [8192,8192,8192],

print_freq_step = 1000, #steps
ckpt_freq_epoch = 5,  #epochs
valid_freq_epoch = 5,  #epochs

checkpoint_dir= "hypotheses/dumbtpth/",
resume_training = False,
)

## --------
parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--load_json', default='configs/bt-train-CFG.json', type=str, metavar='JSON',
    help='Load settings from file in json format. Command line options override values in file.')

args = parser.parse_args()

if args.load_json:
    with open(args.load_json, 'rt') as f:
        CFG.__dict__.update(json.load(f))

### ----------------------------------------------------------------------------
CFG.gLogPath = CFG.checkpoint_dir
CFG.gWeightPath = CFG.checkpoint_dir + '/weights/'

### ============================================================================


def getDataLoaders():

    transform_obj = BarlowTwinsTransformOrig(image_size=CFG.image_size)

    traindataset = Cifar100Dataset( images_folder= os.path.join(CFG.datapath,"train_images"),
                                    csv_path= os.path.join(CFG.datapath,"train_list.csv"),
                                    transform = transform_obj)
    trainloader  = torch.utils.data.DataLoader( traindataset, shuffle=True,
                        batch_size=CFG.batch_size, num_workers=CFG.workers,
                        pin_memory=True)

    validdataset = Cifar100Dataset( images_folder= os.path.join(CFG.datapath,"test_images"),
                                    csv_path= os.path.join(CFG.datapath,"test_list.csv"),
                                    transform = transform_obj)
    validloader  = torch.utils.data.DataLoader( validdataset, shuffle=True,
                        batch_size=CFG.batch_size, num_workers=CFG.workers,
                        pin_memory=True)


    lutl.LOG2DICTXT({"TRAIN DatasetClass":traindataset.get_info(),
                    "TransformsClass": str(transform_obj.get_composition()),
                    }, CFG.gLogPath +'/misc.txt')
    lutl.LOG2DICTXT({"VALID DatasetClass":validdataset.get_info(),
                    "TransformsClass": str(transform_obj.get_composition()),
                    }, CFG.gLogPath +'/misc.txt')

    return trainloader, validloader


def getModelnOptimizer():
    model = BarlowTwins(arch=CFG.featx_arch,  projector=CFG.projector,
                        batch_size=CFG.batch_size,
                        lmbd=CFG.lmbd).to(device)

    optimizer = LARS(model.parameters(), lr=0, weight_decay=CFG.weight_decay,
                     weight_decay_filter=True, lars_adaptation_filter=True)

    model_info = torchsummary.summary(model, 2*[(3, CFG.image_size, CFG.image_size)],
                                verbose=0)
    lutl.LOG2TXT(model_info, CFG.gLogPath +'/misc.txt', console= False)

    return model.to(device), optimizer


def simple_main():
    ### SETUP
    rutl.START_SEED()
    torch.cuda.device(device)
    torch.backends.cudnn.benchmark = True

    if os.path.exists(CFG.checkpoint_dir) and (not CFG.resume_training):
        raise Exception("CheckPoint folder already exists and restart_training not enabled; Somethings Wrong!")
    if not os.path.exists(CFG.gWeightPath): os.makedirs(CFG.gWeightPath)

    with open(CFG.gLogPath+"/exp_config.json", 'a') as f:
        json.dump(vars(CFG), f, indent=4)


    ### DATA ACCESS
    trainloader, validloader = getDataLoaders()

    ### MODEL, OPTIM
    model, optimizer = getModelnOptimizer()

    ## Automatically resume from checkpoint if it exists and enabled
    if os.path.exists(CFG.gWeightPath +'/checkpoint.pth') and CFG.resume_training:
        ckpt = torch.load(CFG.gWeightPath+'/checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lutl.LOG2TXT(f"Restarting Training from EPOCH:{start_epoch} of {CFG.checkpoint_dir}",  CFG.gLogPath +'/misc.txt')
    else:
        start_epoch = 0


    ### MODEL TRAINING
    start_time = time.time()
    best_loss = float('inf')
    if CFG.use_amp: scaler = torch.cuda.amp.GradScaler() # for mixed precision

    for epoch in range(start_epoch, CFG.epochs):

        ## ---- Training Routine ----
        t_running_loss_ = 0
        for step, (y1, y2) in tqdm(enumerate(trainloader, start=epoch * len(trainloader))):
            y1 = y1.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)
            adjust_learning_rate(CFG, optimizer, trainloader, step)
            optimizer.zero_grad()

            if CFG.use_amp: ## with mixed precision
                with torch.cuda.amp.autocast():
                    loss = model.forward(y1, y2)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.forward(y1, y2)
                loss.backward()
                optimizer.step()
            t_running_loss_+=loss.item()

            if step % CFG.print_freq_step == 0:
                stats = dict(epoch=epoch, step=step,
                             lr_weights=optimizer.param_groups[0]['lr'],
                             lr_biases=optimizer.param_groups[1]['lr'],
                             step_loss=loss.item(),
                             time=int(time.time() - start_time))
                lutl.LOG2DICTXT(stats, CFG.checkpoint_dir +'/train-stats.txt')
        train_epoch_loss = t_running_loss_/len(trainloader)

        # save checkpoint
        if (epoch+1) % CFG.ckpt_freq_epoch == 0:
            state = dict(epoch=epoch, model=model.state_dict(),
                            optimizer=optimizer.state_dict())
            torch.save(state, CFG.gWeightPath +'/checkpoint.pth')


        ## ---- Validation Routine ----
        if (epoch+1) % CFG.valid_freq_epoch == 0:
            model.eval()
            v_running_loss_ = 0
            with torch.no_grad():
                for (y1, y2) in tqdm(validloader,  total=len(validloader)):
                    y1 = y1.to(device, non_blocking=True)
                    y2 = y2.to(device, non_blocking=True)
                    loss = model.forward(y1, y2)
                    v_running_loss_ += loss.item()
            valid_epoch_loss = v_running_loss_/len(validloader)
            best_flag = False
            if valid_epoch_loss < best_loss:
                best_flag = True
                torch.save(model.backbone.state_dict(), CFG.gWeightPath +'/encoder-weight.pth')

            v_stats = dict(epoch=epoch, best=best_flag,
                            train_loss=train_epoch_loss,
                            valid_loss=valid_epoch_loss)
            lutl.LOG2DICTXT(v_stats, CFG.gLogPath+'/valid-stats.txt')


if __name__ == '__main__':
    simple_main()