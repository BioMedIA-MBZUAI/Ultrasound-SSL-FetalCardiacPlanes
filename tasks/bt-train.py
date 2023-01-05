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
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl
from algorithms.lars_optim import LARS, adjust_learning_rate
from algorithms.barlows_twin import BarlowTwins
from datacode.imagenet_data import getBarlowTwinDataloader
from datacode.ultrasound_data import getUSBarlowTwinDataloader


print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device Used:", device)

###============================= Configure and Setup ===========================

cfg = rutl.ObjDict(
data= "/home/USR/WERK/data/",
epochs= 1000,
batch_size= 2048,

learning_rate_weights = 0.2,
learning_rate_biases  = 0.0048,
weight_decay = 1e-6,
lambd = 0.0051,
# feature_extract = "resnet18", # "resnet34/50/101"
# featx_pretrain =  "DEFAULT",  # path-to-weights or None
projector = [8192,8192,8192],

print_freq = 100,
checkpoint_dir= "hypotheses/dumbtpth/",
)

## --------
parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--load_json', default='configs/bt-train-cfg.json', type=str, metavar='JSON',
    help='Load settings from file in json format. Command line options override values in file.')

args = parser.parse_args()

if args.load_json:
    with open(args.load_json, 'rt') as f:
        cfg.__dict__.update(json.load(f))

### ----------------------------------------------------------------------------

gLogPath = cfg.checkpoint_dir
gWeightPath = cfg.checkpoint_dir + '/weights/'
if not os.path.exists(gWeightPath): os.makedirs(gWeightPath)

with open(gLogPath+"/exp_bt_cfg.json", 'a') as f:
    json.dump(vars(cfg), f, indent=4)

### ============================================================================


def simple_main():
    rutl.START_SEED()
    gpu = 0
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(cfg).cuda(gpu)
    optimizer = LARS(model.parameters(), lr=0, weight_decay=cfg.weight_decay,
                     weight_decay_filter=True, lars_adaptation_filter=True)

    dataloader,data_info = getUSBarlowTwinDataloader(cfg.data,
                    cfg.batch_size, cfg.workers)
    lutl.LOG2DICTXT(data_info, gLogPath +'/misc.txt')

    ### automatically resume from checkpoint if it exists
    if os.path.exists(gWeightPath +'/checkpoint.pth'):
        ckpt = torch.load(gWeightPath+'/checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lutl.LOG2TXT(f"Restarting Training from EPOCH:{start_epoch} of {cfg.checkpoint_dir}",  gLogPath +'/misc.txt')

    else:
        start_epoch = 0


    ### Training Routine
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler() # for mixed precision
    for epoch in range(start_epoch, cfg.epochs):
        for step, ((y1, y2), _) in tqdm(enumerate(dataloader, start=epoch * len(dataloader))):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(cfg, optimizer, dataloader, step)
            optimizer.zero_grad()
            ## with mixed precision
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()

            if step % cfg.print_freq == 0:
                stats = dict(epoch=epoch, step=step,
                             lr_weights=optimizer.param_groups[0]['lr'],
                             lr_biases=optimizer.param_groups[1]['lr'],
                             loss=loss.item(),
                             time=int(time.time() - start_time))
                lutl.LOG2DICTXT(stats, cfg.checkpoint_dir +'/train-stats.txt')

        # save checkpoint
        state = dict(epoch=epoch + 1, model=model.state_dict(),
                        optimizer=optimizer.state_dict())
        torch.save(state, gWeightPath +'/checkpoint.pth')
    # save final model
    torch.save(model.backbone.state_dict(), gWeightPath +'/encoder-weight.pth')


if __name__ == '__main__':
    simple_main()