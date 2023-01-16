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
from algorithms.classifiers import ClassifierNet
from datacode.ultrasound_data import getUSClassifyDataloader
from datacode.general_data import getCifar100Dataloader


print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device Used:", device)

###============================= Configure and Setup ===========================
cfg = rutl.ObjDict(
train_data= "/home/USR/WERK/data/train",
valid_data = "/home/USR/WERK/data/valid",
epochs= 1000,
batch_size= 2048,
workers= 4,
balance_class = True,
augument_list = ["hflp", "vflp"], #["crop", "cntr", "brig", "affn", "pers", "hflp", "vflp"]

learning_rate= 0.2,
weight_decay= 1e-6,
feature_extract = "resnet18", # "resnet34/50/101"
featx_pretrain =  "DEFAULT",  # path-to-weights or None
featx_dropout = 0,
featx_freeze =  False,

classifier = [1024, 6],
clsfy_dropout = 0.5,

checkpoint_dir= "hypotheses/dummypth/",
restart_training=False
)

### -----
parser = argparse.ArgumentParser(description='Classification task')
parser.add_argument('--load_json', default='configs/cls-train-cfg.json', type=str, metavar='JSON',
    help='Load settings from file in json format. Command line options override values in file.')

args = parser.parse_args()

if args.load_json:
    with open(args.load_json, 'rt') as f:
        cfg.__dict__.update(json.load(f))

### ----------------------------------------------------------------------------
cfg.gLogPath = cfg.checkpoint_dir
cfg.gWeightPath = cfg.checkpoint_dir + '/weights/'

### ============================================================================


def simple_main():

    ### SETUP
    rutl.START_SEED()
    gpu = 0
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    if os.path.exists(cfg.checkpoint_dir) and (not cfg.restart_training):
        raise Exception("CheckPoint folder already exists and restart_training not enabled; Somethings Wrong!")
    if not os.path.exists(cfg.gWeightPath): os.makedirs(cfg.gWeightPath)

    with open(cfg.gLogPath+"/exp_cfg.json", 'a') as f:
        json.dump(vars(cfg), f, indent=4)


    ### MODEL, OPTIM
    model = ClassifierNet(cfg).cuda(gpu)
    lossfn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                        weight_decay=cfg.weight_decay)
    lutl.LOG2TXT(f"Parameters:{rutl.count_train_param(model)}", cfg.gLogPath +'/misc.txt')

    ## Automatically resume from checkpoint if it exists and enabled
    if os.path.exists(cfg.gWeightPath +'/checkpoint.pth'):
        ckpt = torch.load(cfg.gWeightPath+'/checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lutl.LOG2TXT(f"Restarting Training from EPOCH:{start_epoch} of {cfg.checkpoint_dir}",  cfg.gLogPath +'/misc.txt')
    else:
        start_epoch = 0


    ### DATA ACCESS
    trainloader, data_info = getUSClassifyDataloader(cfg.train_data,
                        cfg.batch_size, cfg.workers,  type_='train',
                        augument_list=cfg.augument_list,
                        balance_class=cfg.balance_class )
    lutl.LOG2DICTXT(data_info, cfg.gLogPath +'/misc.txt')
    validloader, data_info = getUSClassifyDataloader(cfg.valid_data,
                        cfg.batch_size, cfg.workers, type_='valid')
    lutl.LOG2DICTXT(data_info, cfg.gLogPath +'/misc.txt')


    ### MODEL TRAINING
    start_time = time.time()
    best_acc = 0 ; best_loss = float('inf')
    trainMetric = MultiClassMetrics(cfg.gLogPath)
    validMetric = MultiClassMetrics(cfg.gLogPath)
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
            ## END with
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            trainMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss)

        ## save checkpoint states
        state = dict(epoch=epoch + 1, model=model.state_dict(),
                        optimizer=optimizer.state_dict())
        torch.save(state, cfg.gWeightPath +'/checkpoint.pth')


        ## ---- Validation Routine ----
        model.eval()
        with torch.no_grad():
            for img, tgt in tqdm(validloader):
                img = img.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                # with torch.cuda.amp.autocast():
                pred = model.forward(img)
                loss = lossfn(pred, tgt)
                ## END with
                validMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss)

        ## Log Metrics
        stats = dict( epoch=epoch, time=int(time.time() - start_time),
                    trainloss = trainMetric.get_loss(),
                    trainacc = trainMetric.get_accuracy(),
                    validloss = validMetric.get_loss(),
                    validacc = validMetric.get_accuracy(), )
        lutl.LOG2DICTXT(stats, cfg.gLogPath+'/train-stats.txt')


        ## save best model
        best_flag = False
        if stats['validacc'] > best_acc:
            torch.save(model.state_dict(), cfg.gWeightPath +'/bestmodel.pth')
            best_acc = stats['validacc']
            best_loss = stats['validloss']
            best_flag = True

        ## Log detailed validation
        detail_stat = dict( epoch=epoch, time=int(time.time() - start_time),
                            best = best_flag,
                            validreport =  validMetric.get_class_report() )
        lutl.LOG2DICTXT(detail_stat, cfg.gLogPath+'/validation-details.txt', console=False)

        trainMetric.reset()
        validMetric.reset(best_flag)

if __name__ == '__main__':
    simple_main()