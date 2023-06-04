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
import torchinfo
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl
from algorithms.barlowtwins import BarlowTwins, LARS, adjust_learning_rate
from algorithms.loss.supcon_loss import SupConLoss

from datacode.natural_image_data import Cifar100Dataset
from datacode.ultrasound_data import FetalUSFramesDataset, ClassifyDataFromCSV
from datacode.augmentations import BarlowTwinsTransformOrig, ClassifierTransform


print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device Used:", device)

###============================= Configure and Setup ===========================

CFG = rutl.ObjDict(
use_amp = True, #automatic Mixed precision

datapath    = "/home/USR/WERK/data/a.hdf5",
valdatapath = "/home/USR/WERK/valdata/b.hdf5",
supdatafolder = "/home/USR/WERK/classdata/",

skip_count = 5,
epochs      = 1000,
batch_size  = 2048,

learning_rate_weights = 0.2,
learning_rate_biases  = 0.0048,
weight_decay = 1e-6,
lmbd = 0.0051,
image_size=256,

featx_arch = "resnet50", # "resnet34/50/101"
featx_pretrain =  None, # "IMGNET-1K" or None
projector = [8192,8192,8192],

print_freq_step = 1000, #steps
ckpt_freq_epoch = 5,  #epochs
valid_freq_epoch = 5,  #epochs
disable_tqdm=False,   #True--> to disable

checkpoint_dir= "hypotheses/-dummy/ssl-barlow/",
resume_training = False,
)

## --------
parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--load-json', type=str, metavar='JSON',
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
    """ Unlabelled SSL Dataset
    """

    transform_obj = BarlowTwinsTransformOrig(image_size=CFG.image_size)

    traindataset = FetalUSFramesDataset( hdf5_file= CFG.datapath,
                                transform = transform_obj,
                                load2ram = False, frame_skip=CFG.skip_count)


    trainloader  = torch.utils.data.DataLoader( traindataset, shuffle=True,
                        batch_size=CFG.batch_size, num_workers=CFG.workers,
                        pin_memory=True)

    validdataset = FetalUSFramesDataset( hdf5_file= CFG.valdatapath,
                                transform = transform_obj,
                                load2ram = False, frame_skip=CFG.skip_count)


    validloader  = torch.utils.data.DataLoader( validdataset, shuffle=False,
                        batch_size=CFG.batch_size, num_workers=CFG.workers,
                        pin_memory=True)


    lutl.LOG2DICTXT({"TRAIN DatasetClass":traindataset.get_info(),
                    "TransformsClass": str(transform_obj.get_composition()),
                    }, CFG.gLogPath +'/misc.txt')
    lutl.LOG2DICTXT({"VALID DatasetClass":validdataset.get_info(),
                    "TransformsClass": str(transform_obj.get_composition()),
                    }, CFG.gLogPath +'/misc.txt')

    return trainloader, validloader


def getSupConDataIter():
    """ Get labelled Classification dataset domain
    """

    data_percent = 1
    batch_size = 32

    transform_obj = BarlowTwinsTransformOrig(image_size=CFG.image_size)
    traindataset = ClassifyDataFromCSV(CFG.supdatafolder,
                                       CFG.supdatafolder+"/trainV3.csv",
                                       transform = transform_obj,)

    _idx, used_idx = sk_train_test_split( np.arange(len(traindataset)),
                            test_size=data_percent/100, random_state=1729,
                            stratify=traindataset.targets)
    traindataset = torch.utils.data.Subset(traindataset, sorted(used_idx))
    lutl.LOG2CSV(sorted(used_idx), CFG.gLogPath +'/ClassifyTrain_indices_used.csv')

    trainloader  = torch.utils.data.DataLoader( traindataset, shuffle=True,
                    batch_size=batch_size, num_workers=CFG.workers,
                    pin_memory=True)

    global step_loader
    step_loader = iter(trainloader)

    def data_next():
        global step_loader
        try: step_data = next(step_loader)
        except StopIteration:
            step_loader = iter(trainloader)
            step_data = next(step_loader)
        return step_data

    return data_next




def getModelnOptimizer():
    model = BarlowTwins(featx_arch=CFG.featx_arch,
                        projector_sizes=CFG.projector,
                        batch_size=CFG.batch_size,
                        lmbd=CFG.lmbd,
                        pretrained=CFG.featx_pretrain).to(device)

    optimizer = LARS(model.parameters(), lr=0, weight_decay=CFG.weight_decay,
                     weight_decay_filter=True, lars_adaptation_filter=True)

    model_info = torchinfo.summary(model, 2*[(1, 3, CFG.image_size, CFG.image_size)],
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

    ### Sup-Con setup
    supcon_datanext = getSupConDataIter()
    supcon_loss  = SupConLoss(temperature=1)
    supcon_layer = nn.Sequential( nn.Linear(model.outfeatx_size, 512),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(512, 128)).to(device)

    ## Automatically resume from checkpoint if it exists and enabled
    ckpt = None
    if CFG.resume_training:
        try:    ckpt = torch.load(CFG.gWeightPath+'/checkpoint-1.pth', map_location='cpu')
        except:
            try:ckpt = torch.load(CFG.gWeightPath+'/checkpoint-0.pth', map_location='cpu')
            except: print("Check points are not loadable. Starting fresh...")
    if ckpt:
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lutl.LOG2TXT(f"Restarting Training from EPOCH:{start_epoch} of {CFG.checkpoint_dir}",  CFG.gLogPath +'/misc.txt')
    else:
        start_epoch = 0


    ### MODEL TRAINING
    start_time = time.time()
    best_loss = float('inf')
    wgt_suf   = 0  # foolproof savetime crash
    if CFG.use_amp: scaler = torch.cuda.amp.GradScaler() # for mixed precision

    for epoch in range(start_epoch, CFG.epochs):

        ## ---- Training Routine ----
        t_running_loss_ = 0
        model.train()
        for step, (y1, y2) in tqdm(enumerate(trainloader,
                                    start=epoch * len(trainloader)),
                                    disable=CFG.disable_tqdm):
            y1 = y1.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)

            ## sup-con data
            ((sc_img1,sc_img2), sc_tgt) = supcon_datanext()
            sc_img1 = sc_img1.to(device, non_blocking=True)
            sc_img2 = sc_img2.to(device, non_blocking=True)
            sc_tgt = sc_tgt.to(device, non_blocking=True)

            adjust_learning_rate(CFG, optimizer, trainloader, step)
            optimizer.zero_grad()

            if CFG.use_amp: ## with mixed precision
                with torch.cuda.amp.autocast():
                    bt_loss = model.forward(y1, y2)
                    ## sup-con step
                    sc_pred1 = supcon_layer(model.backbone(sc_img1))
                    sc_pred2 = supcon_layer(model.backbone(sc_img2))
                    sc_pred = torch.cat(( sc_pred1.unsqueeze(1),
                                          sc_pred2.unsqueeze(1) ),1)
                    sc_loss = supcon_loss(sc_pred, sc_tgt)

                loss = bt_loss + sc_loss
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
                             lr_weights  = optimizer.param_groups[0]['lr'],
                             lr_biases   = optimizer.param_groups[1]['lr'],
                             step_loss   = loss.item(),
                             barlow_loss = bt_loss.item(),
                             supcon_loss = sc_loss.item(),
                             time=int(time.time() - start_time))
                lutl.LOG2DICTXT(stats, CFG.checkpoint_dir +'/train-stats.txt')
        train_epoch_loss = t_running_loss_/len(trainloader)

        # save checkpoint
        if (epoch+1) % CFG.ckpt_freq_epoch == 0:
            wgt_suf = (wgt_suf+1) %2
            state = dict(epoch=epoch, model=model.state_dict(),
                            optimizer=optimizer.state_dict())
            torch.save(state, CFG.gWeightPath +f'/checkpoint-{wgt_suf}.pth')


        ## ---- Validation Routine ----
        if (epoch+1) % CFG.valid_freq_epoch == 0:
            model.eval()
            v_running_loss_ = 0
            with torch.no_grad():
                for (y1, y2) in tqdm(validloader,  total=len(validloader),
                                    disable=CFG.disable_tqdm):
                    y1 = y1.to(device, non_blocking=True)
                    y2 = y2.to(device, non_blocking=True)
                    loss = model.forward(y1, y2)
                    v_running_loss_ += loss.item()
            valid_epoch_loss = v_running_loss_/len(validloader)

            # just check
            best_flag = False
            if valid_epoch_loss < best_loss:
                best_flag = True
                best_loss = valid_epoch_loss
                # torch.save(model.backbone.state_dict(), CFG.gWeightPath +f'/encoder-weight-{wgt_suf}.pth')

            v_stats = dict(epoch=epoch, best=best_flag, wgt_suf=wgt_suf,
                            train_loss=train_epoch_loss,
                            valid_loss=valid_epoch_loss)
            lutl.LOG2DICTXT(v_stats, CFG.gLogPath+'/valid-stats.txt')


if __name__ == '__main__':
    simple_main()