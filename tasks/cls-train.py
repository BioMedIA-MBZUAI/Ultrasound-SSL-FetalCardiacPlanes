""" Classifier Network trainig
"""

import argparse
import json
import os
import sys
import time
from tqdm.autonotebook import tqdm

import torch
from torch import nn, optim
import torchinfo

import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl
from utilities.metricUtils import MultiClassMetrics
from algorithms.classifiers import ClassifierNet
from datacode.ultrasound_data import ClassifyDataFromCSV, get_class_weights
from datacode.augmentations import ClassifierTransform

print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device Used:", device)

###============================= Configure and Setup ===========================

CFG = rutl.ObjDict(
data_folder  = "/home/joseph.benjamin/WERK/fetal-ultrasound/data/Fetal-UltraSound/US-Planes-Heart-Views-V3",
balance_data = False, #while loading in dataloader; removed
seed = 1792,  #previously 73

epochs        = 100,
image_size    = 256,
batch_size    = 128,
workers       = 16,
learning_rate = 1e-3,
weight_decay  = 1e-6,

featx_arch     = "resnet50",
featx_pretrain =  "IMAGENET-1K" , # "IMAGENET-1K" or None
featx_freeze   = False,
featx_bnorm    = False,
featx_dropout  = 0.5,
clsfy_layers   = [5], #First mlp inwill be set w.r.t FeatureExtractor
clsfy_dropout  = 0.0,

checkpoint_dir   = "hypotheses/#dummy/Classify/trail-002",
disable_tqdm     = False, #True--> to disable
restart_training = True
)

### ----------------------------------------------------------------------------
# CLI TAKES PRECENCE OVER JSON CONFIG
# e.g CLI overwrites the value set for featx-pretain in JSON while running
# without CLI default values form dict will be used

parser = argparse.ArgumentParser(description='Classification task')
parser.add_argument('--load-json', type=str, metavar='JSON',
    help='Load settings from file in json format. Command line options override values in file.')

parser.add_argument('--seed', type=int, metavar='INT',
    help='add batchnorm between feature extractor and classifier')

parser.add_argument('--featx-freeze', type=bool, metavar='BOOL',
    help='freeze pretrain or not')

parser.add_argument('--featx-bnorm', type=bool, metavar='BOOL',
    help='add batchnorm between feature extractor and classifier')

parser.add_argument('--featx-pretrain', type=str, metavar='PATH',
    help='Set from where to load the prestrained weight from')

parser.add_argument('--checkpoint-dir', type=str, metavar='PATH',
    help='Load settings from file in json format. Command line options override values in file.')


args = parser.parse_args()

if args.load_json:
    with open(args.load_json, 'rt') as f:
        CFG.__dict__.update(json.load(f))

for arg in vars(args):
    att = getattr(args, arg)
    if att: CFG.__dict__[arg] = att

### ----------------------------------------------------------------------------
CFG.gLogPath = CFG.checkpoint_dir
CFG.gWeightPath = CFG.checkpoint_dir + '/weights/'

### ============================================================================

def getDataLoaders(data_percent=None):
    ## Augumentations
    train_transforms =ClassifierTransform(image_size=CFG.image_size, mode="train")
    valid_transforms =ClassifierTransform(image_size=CFG.image_size, mode="infer")

    ## Dataset Class
    traindataset = ClassifyDataFromCSV(CFG.data_folder,
                                       CFG.data_folder+"/trainV3.csv",
                                       transform = train_transforms,)
    validdataset = ClassifyDataFromCSV(CFG.data_folder,
                                       CFG.data_folder+"/validV3.csv",
                                       transform = valid_transforms,)
    class_weights, _ = get_class_weights(traindataset.targets, nclasses=5)

    ### Choose P% of data from train data
    if data_percent and (data_percent < 100):
        _idx, used_idx = sk_train_test_split( np.arange(len(traindataset)),
                                test_size=data_percent/100, random_state=CFG.seed,
                                stratify=traindataset.targets)
        traindataset = torch.utils.data.Subset(traindataset, sorted(used_idx))
        lutl.LOG2CSV(sorted(used_idx), CFG.gLogPath +'/train_indices_used.csv')

    torch.manual_seed(CFG.seed)
    ## Loaders Class
    trainloader  = torch.utils.data.DataLoader( traindataset, shuffle=True,
                        batch_size=CFG.batch_size, num_workers=CFG.workers,
                        pin_memory=True)

    validloader  = torch.utils.data.DataLoader( validdataset, shuffle=False,
                        batch_size=CFG.batch_size, num_workers=CFG.workers,
                        pin_memory=True)

    lutl.LOG2DICTXT({"Train->":len(traindataset),
                    "class-weights":str(class_weights),
                    "TransformsClass": str(train_transforms.get_composition()),
                    },CFG.gLogPath +'/misc.txt')
    lutl.LOG2DICTXT({"Valid->":len(validdataset),
                    "TransformsClass": str(valid_transforms.get_composition()),
                    },CFG.gLogPath +'/misc.txt')

    return trainloader, validloader, class_weights


def getModelnOptimizer():

    ## pretrain setting
    m_state = 0; torch_pretrain_flag = None
    if os.path.isfile(CFG.featx_pretrain):
        m_state = torch.load(CFG.featx_pretrain, map_location='cpu')
    else: torch_pretrain_flag = CFG.featx_pretrain

    model = ClassifierNet(arch=CFG.featx_arch,
                    fc_layer_sizes=CFG.clsfy_layers,
                    feature_freeze=CFG.featx_freeze,
                    feature_dropout=CFG.featx_dropout,
                    feature_bnorm=CFG.featx_bnorm,
                    classifier_dropout=CFG.clsfy_dropout,
                    torch_pretrain=torch_pretrain_flag )

    ## load from checkpoints
    if m_state:
        m_state = m_state["model"]
        ret_msg = model.load_state_dict(m_state, strict=False)
        lutl.LOG2TXT(f"Manual Pretrain Loaded...{CFG.featx_pretrain},{str(ret_msg)}",
                     CFG.gLogPath +'/misc.txt')

    model_info = torchinfo.summary(model, (1, 3, CFG.image_size, CFG.image_size),
                                verbose=0)
    lutl.LOG2TXT(model_info, CFG.gLogPath +'/misc.txt', console= False)

    ##--------------

    optimizer = optim.AdamW(model.parameters(), lr=CFG.learning_rate,
                        weight_decay=CFG.weight_decay)
    scheduler = False

    return model.to(device), optimizer, scheduler


def getLossFunc(class_weights):
    lossfn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights,
                                        dtype=torch.float32).to(device) )
    return lossfn


def simple_main(data_percent=None):

   ### SETUP
    rutl.START_SEED(CFG.seed)
    gpu = 0
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    ## paths and logs setup
    if data_percent: CFG.gLogPath = CFG.checkpoint_dir+f"/{data_percent}_percent/"
    CFG.gWeightPath = CFG.gLogPath+"/weights/"

    if os.path.exists(CFG.gLogPath) and (not CFG.restart_training):
        raise Exception("CheckPoint folder already exists and restart_training not enabled; Somethings Wrong!",
                        CFG.checkpoint_dir)
    if not os.path.exists(CFG.gWeightPath): os.makedirs(CFG.gWeightPath)

    with open(CFG.gLogPath+"/exp_cfg.json", 'a') as f:
        json.dump(vars(CFG), f, indent=4)


    ### DATA ACCESS
    trainloader, validloader, class_weights  = getDataLoaders(data_percent)

    ### MODEL, OPTIM
    model, optimizer, scheduler = getModelnOptimizer()
    lossfn = getLossFunc(class_weights)


    ## Automatically resume from checkpoint if it exists and enabled
    if os.path.exists(CFG.gWeightPath +'/checkpoint.pth') and CFG.restart_training:
        ckpt = torch.load(CFG.gWeightPath  +'/checkpoint.pth',
                            map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lutl.LOG2TXT(f"Restarting Training from EPOCH:{start_epoch} of {CFG.gLogPath}",  CFG.gLogPath +'/misc.txt')
    else:
        start_epoch = 0

    ### MODEL TRAINING
    start_time = time.time()
    best_acc = 0 ; best_loss = float('inf')
    trainMetric = MultiClassMetrics(CFG.gLogPath)
    validMetric = MultiClassMetrics(CFG.gLogPath)

    for epoch in range(start_epoch, CFG.epochs):

        ## ---- Training Routine ----
        model.train()
        for img, tgt in tqdm(trainloader, disable=CFG.disable_tqdm):
            img = img.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model.forward(img)
            loss = lossfn(pred, tgt)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(),
            #                          max_norm=2.0, norm_type=2)
            optimizer.step()
            trainMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss)
        if scheduler: scheduler.step()

        ## save checkpoint states
        state = dict(epoch=epoch + 1, model=model.state_dict(),
                        optimizer=optimizer.state_dict())
        torch.save(state, CFG.gWeightPath +'/checkpoint.pth')


        ## ---- Validation Routine ----
        model.eval()
        with torch.no_grad():
            for img, tgt in tqdm(validloader, disable=CFG.disable_tqdm):
                img = img.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                pred = model.forward(img)
                loss = lossfn(pred, tgt)
                validMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss)

        ## Log Metrics TODO Add balanced and F1
        stats = dict(
                epoch=epoch, time=int(time.time() - start_time),
                trainloss = trainMetric.get_loss(),
                trainacc  = trainMetric.get_balanced_accuracy(),
                trainF1   = trainMetric.get_f1score(),
                validloss = validMetric.get_loss(),
                validacc  = validMetric.get_balanced_accuracy(),
                validF1   = validMetric.get_f1score(),
                )
        lutl.LOG2DICTXT(stats, CFG.gLogPath+'/train-stats.txt')


        ## save best model
        best_flag = False
        if stats['validacc'] > best_acc:
            torch.save(model.state_dict(), CFG.gWeightPath +'/bestmodel.pth')
            best_acc = stats['validacc']
            best_loss = stats['validloss']
            best_flag = True

        ## Log detailed validation
        detail_stat = dict(
                epoch=epoch, time=int(time.time() - start_time),
                best = best_flag,
                validf1scr  = validMetric.get_f1score(),
                validbalacc = validMetric.get_balanced_accuracy(),
                validacc    = validMetric.get_accuracy(),
                validreport = validMetric.get_class_report(),
                validconfus = validMetric.get_confusion_matrix().tolist(),
            )
        lutl.LOG2DICTXT(detail_stat, CFG.gLogPath+'/validation-details.txt', console=False)

        trainMetric.reset()
        validMetric.reset(best_flag)

    return CFG.gLogPath



def simple_test(saved_logpath):

    ### SETUP
    rutl.START_SEED()
    gpu = 0
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    ### DATA ACCESS
    test_transforms =ClassifierTransform(image_size=CFG.image_size,
                                        mode="infer")
    testdataset = ClassifyDataFromCSV(  CFG.data_folder,
                                        CFG.data_folder+"/testV3.csv",
                                        transform = test_transforms,)
    testloader  = torch.utils.data.DataLoader( testdataset,
                                        shuffle=False,
                                        batch_size=CFG.batch_size,
                                        num_workers=CFG.workers,
                                        pin_memory=True)
    lutl.LOG2DICTXT({"TEST->":len(testdataset),
                     "TransformsClass": str(test_transforms.get_composition()),
                    },saved_logpath +'/test-results.txt')

    ### MODEL
    model = ClassifierNet(arch=CFG.featx_arch,
                    fc_layer_sizes=CFG.clsfy_layers,
                    feature_freeze=CFG.featx_freeze,
                    feature_dropout=CFG.featx_dropout,
                    feature_bnorm=CFG.featx_bnorm,
                    classifier_dropout=CFG.clsfy_dropout)
    model = model.to(device)
    model.load_state_dict(torch.load(saved_logpath+"/weights/bestmodel.pth"))


    ### MODEL TESTING
    testMetric = MultiClassMetrics(saved_logpath)
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        for img, tgt in tqdm(testloader, disable=CFG.disable_tqdm):
            img = img.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            pred = model.forward(img)
            testMetric.add_entry(torch.argmax(pred, dim=1), tgt)

        ## Log detailed validation
        detail_stat = dict(
                timetaken   = int(time.time() - start_time),
                testf1scr  = testMetric.get_f1score(),
                testbalacc = testMetric.get_balanced_accuracy(),
                testacc    = testMetric.get_accuracy(),
                testreport = testMetric.get_class_report(),
                testconfus = testMetric.get_confusion_matrix(
                                        save_png= True, title="test").tolist(),
            )
        lutl.LOG2DICTXT(detail_stat, saved_logpath+'/test-results.txt',
                        console=True)

        testMetric._write_predictions(title="test")



if __name__ == '__main__':

    # logpth = simple_main()
    # simple_test(logpth)

    for p in [100, 50, 25, 10, 5, 1]:
        logpth = simple_main(data_percent=p)
        simple_test(logpth)