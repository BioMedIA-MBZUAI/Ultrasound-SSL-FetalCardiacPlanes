import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
from algorithms.lars_optim import LARS, adjust_learning_rate
from algorithms.barlows_twin import BarlowTwins
from datacode.imagenet_data import classify_dataloader


print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device Used:", device)

###============================= Configure and Setup ===========================

parser = argparse.ArgumentParser(description='Barlow Twins Training')

parser.add_argument('--data', default="/home/joseph.benjamin/WERK/data/cifar-100-python/",
                    type=str, metavar='DATADIR',
                    help='path to dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=1024, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./hypotheses/exp_temp/', type=str,
                    metavar='DIR', help='path to checkpoint directory')
### -----
parser.add_argument('--load_json', default='hypotheses/config/bt_train_cfg.json', type=str, metavar='DIR',
    help='Load settings from file in json format. Command line options override values in file.')

args = parser.parse_args()

if args.load_json:
    with open(args.load_json, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

### ----------------------------------------------------------------------------

LOGPATH = args.checkpoint_dir
WGTPATH = LOGPATH + '/weights/'
if not os.path.exists(WGTPATH): os.makedirs(WGTPATH)

with open(LOGPATH+"/exp_cfg.json", 'wt') as f:
    json.dump(vars(args), f, indent=4)

### ============================================================================


def simple_main():
    rutl.START_SEED()
    gpu = 0
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    stats_file = open(args.checkpoint_dir +'/stats.txt', 'a', buffering=1)

    model = BarlowTwins(args).cuda(gpu)

    optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    if os.path.exists(args.checkpoint_dir +'/checkpoint.pth'):
        ckpt = torch.load(args.checkpoint_dir+'/checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0


    loader = classify_dataloader(args.data, args.batch_size, args.workers)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler() # for mixed precision
    for epoch in range(start_epoch, args.epochs):
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            ## with mixed precision
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                stats = dict(epoch=epoch, step=step,
                                lr_weights=optimizer.param_groups[0]['lr'],
                                lr_biases=optimizer.param_groups[1]['lr'],
                                loss=loss.item(),
                                time=int(time.time() - start_time))
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
        # save checkpoint
        state = dict(epoch=epoch + 1, model=model.state_dict(),
                        optimizer=optimizer.state_dict())
        torch.save(state, args.checkpoint_dir +'/checkpoint.pth')
    # save final model
    torch.save(model.module.backbone.state_dict(),
                args.checkpoint_dir +'/resnet50.pth')



if __name__ == '__main__':
    simple_main()