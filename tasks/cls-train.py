import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

from tqdm.autonotebook import tqdm
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl
from utilities.metricUtils import MultiClassMetrics
from algorithms.lars_optim import LARS, adjust_learning_rate
from algorithms.resnet import ClassifierNet
from datacode.ctscan_data import CT_classify_dataloader
from datacode.general_data import getCifar100Dataloader


print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device Used:", device)

###============================= Configure and Setup ===========================
cfg = rutl.ObjDict(
data= "/home/USR/WERK/data/",
epochs= 1000,
batch_size= 2048,
learning_rate= 0.2,
weight_decay= 1e-6,
classifier= [1024, 6],
checkpoint_dir= "hypotheses/dummypth/",
)

### -----
parser = argparse.ArgumentParser(description='Classification task')
parser.add_argument('--load_json', default='hypotheses/config/cls-train-cfg.json', type=str, metavar='JSON',
    help='Load settings from file in json format. Command line options override values in file.')

args = parser.parse_args()

if args.load_json:
    with open(args.load_json, 'rt') as f:
        cfg.__dict__.update(json.load(f))

### ----------------------------------------------------------------------------

gLogPath = cfg.checkpoint_dir
gWeightPath = cfg.checkpoint_dir + '/weights/'
if not os.path.exists(gWeightPath): os.makedirs(gWeightPath)

with open(gLogPath+"/exp_cfg.json", 'wt') as f:
    json.dump(vars(cfg), f, indent=4)

### ============================================================================



def simple_main():
    rutl.START_SEED()
    gpu = 0
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = ClassifierNet(cfg).cuda(gpu)
    lossfn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                        weight_decay=cfg.weight_decay)
    lutl.LOG2TXT(f"Parameters:{rutl.count_train_param(model)}", gLogPath +'/misc.txt')


    trainloader, cls_idx = getCifar100Dataloader(cfg.data,
                        cfg.batch_size, cfg.workers, type_='train' )
    validloader, _ = getCifar100Dataloader(cfg.data,
                        cfg.batch_size, cfg.workers, type_='valid')
    lutl.LOG2DICTXT(cls_idx, gLogPath +'/misc.txt')

    ## Automatically resume from checkpoint if it exists
    if os.path.exists(gWeightPath +'/checkpoint.pth'):
        ckpt = torch.load(gWeightPath+'/checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Restarting Training from EPOCH:{start_epoch} of {cfg.checkpoint_dir}")
    else:
        start_epoch = 0


    ### MODEL TRAINING
    start_time = time.time()
    best_acc = 0 ; best_loss = float('inf')
    trainMetric = MultiClassMetrics(); validMetric = MultiClassMetrics()
    scaler = torch.cuda.amp.GradScaler() # for mixed precision
    for epoch in range(start_epoch, cfg.epochs):

        ## ---- Training Routine ----
        model.train()
        for img, tgt in tqdm(trainloader):
            img = img.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            optimizer.zero_grad()
            ## with mixed precision
            with torch.cuda.amp.autocast():
                pred = model.forward(img)
                loss = lossfn(pred, tgt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            trainMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss)

        ## save checkpoint states
        state = dict(epoch=epoch + 1, model=model.state_dict(),
                        optimizer=optimizer.state_dict())
        torch.save(state, gWeightPath +'/checkpoint.pth')

        ## ---- Validation Routine ----

        model.eval()
        with torch.no_grad():
            for img, tgt in tqdm(validloader):
                img = img.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    pred = model.forward(img)
                    loss = lossfn(pred, tgt)
                validMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss)

        ## Log Metrics
        stats = dict( epoch=epoch, time=int(time.time() - start_time),
                    trainloss = trainMetric.get_loss(),
                    trainacc = trainMetric.get_accuracy(),
                    validloss = validMetric.get_loss(),
                    validacc = validMetric.get_accuracy(), )
        lutl.LOG2DICTXT(stats, gLogPath+'/train-stats.txt')
        detail_stat = dict( epoch=epoch, time=int(time.time() - start_time),
                            validreport =  validMetric.get_class_report() )
        lutl.LOG2DICTXT(detail_stat, gLogPath+'/validation-details.txt', console=False)

        ## save best model
        if stats['validacc'] > best_acc:
            torch.save(model.state_dict(), gWeightPath +'/bestmodel.pth')
            best_acc = stats['validacc']
            best_loss = stats['validloss']

        trainMetric.reset()
        validMetric.reset()

if __name__ == '__main__':
    simple_main()