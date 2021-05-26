from utils import save_checkpoint, AverageMeter, accuracy
import argparse
import os
import time
from itertools import combinations
import VGG19_model as VGG19
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import warnings
from torch.optim import lr_scheduler
import torch.nn as nn
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--epochs', default=130, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=150, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--Trainedresume', default='', type=str, metavar='Path of pretrained model')
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--lr_decay_ratio', type=float, default=0.2)
parser.add_argument('--epoch_drop', nargs='*', type=int, default=(60, 120, 160))
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of Jpeg loading workers (default: 4)')
parser.add_argument('-Lambda', default=0,help='number of Jpeg loading workers (default: 4)')
parser.add_argument('-Ratio',type=float,help='Percentages of encrypted parameters')
parser.add_argument('-Repeat',type=int, help='Repeat times')
parser.add_argument('-layers',type=list, help = 'layers to be encrypted')
parser.add_argument('-importance_resume',type=str, help = 'Path to save importance of parameters')

parser.set_defaults(Repeat=20)
parser.set_defaults(layers=[0,1,4,8])
parser.set_defaults(Lambda= 0.0001)
parser.set_defaults(importance_resume='Importance')
parser.set_defaults(Trainedresume='PretrainedModel/VGG19.pth.tar')
best_prec1 = 100
writer = None
time_acc = [(0, 0, 0)]
total_steps = 0
exp_flops, exp_l0 = [], []


def main():
    global args, best_prec1, writer, time_acc, total_steps, exp_flops, exp_l0
    args = parser.parse_args()
    #### Load Data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./Jpeg', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    ## Select
    for r in range(1,21):
        args.Ratio = 0.02*r
        print('#################Percentage: {}#################'.format(args.Ratio))
        print('######## Encrypt layers: {} #########'.format(args.layers))
        ## Select Important Parameters
        ES = Select(args)
        for method in ES.keys():
            ACs = 0
            for SEED in np.random.permutation(200)[0:args.Repeat]:
                Es = ES[method]
                ## Load Pretrained Model
                model = VGG19.vgg19()
                model.features = torch.nn.DataParallel(model.features)
                checkpoint = torch.load(args.Trainedresume)
                model.load_state_dict(checkpoint['state_dict'])
                ## Encrypt
                model = Encrypt(model,Es,SEED)
                AC = validate(val_loader, model)
                ACs = ACs+AC
            print('The Accuracy of the {} is {}'.format(method,ACs/args.Repeat))



def Select(args):
    Model = VGG19.vgg19()
    Model.features = torch.nn.DataParallel(Model.features)

    for layerID in args.layers:
        path = os.path.join(args.importance_resume, 'VGG19_layer_{}.pth.tar'.format(layerID))
        if not os.path.exists(path):
            print('Start to learn the importance of the {}-th layer'.format(layerID))
            Get_Importance(layerID,args)
    ES = {}
    for method in ['PSS','Random','Descend','Ascend','Mean']:
        Es = {}
        for layerID in args.layers:
            path = os.path.join(args.importance_resume,'VGG19_layer_{}.pth.tar'.format(layerID))
            checkpoint = torch.load(path)
            importance = checkpoint['importance']
            Model.layers[layerID].qz_loga.data.copy_(importance)
            E = Model.layers[layerID].EncryptLocation(args.Ratio, method)
            Es[layerID] = E
        ES[method] = Es
    return ES

def Get_Importance(layerID,args):
    ## Data loading
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    Dataset = datasets.CIFAR10(root='./Jpeg', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    Dataset.targets = Dataset.targets[0:30000]
    Dataset.data = Dataset.data[0:30000]
    train_loader = torch.utils.data.DataLoader(Dataset
        ,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # create model
    model = VGG19.vgg19()
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # Load pretrained model
    if os.path.isfile(args.Trainedresume):
        print("=> loading checkpoint '{}'".format(args.Trainedresume))
        checkpoint = torch.load(args.Trainedresume)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.Trainedresume))

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, nesterov=True)

    loglike = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loglike = loglike.cuda()

    def loss_function(model,output, target_var,layerID,Lambda):
        loss = loglike(output, target_var)
        reg = model.layers[layerID].regularization()
        total_loss = loss + Lambda*reg
        if torch.cuda.is_available():
            total_loss = total_loss.cuda()
        return total_loss,loss,Lambda*reg

    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=args.epoch_drop, gamma=args.lr_decay_ratio)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for i,layer in enumerate(model.layers):
        if hasattr(layer,'qz_loga'):
            layer.reset_to_10()
            if i == layerID:
                layer.qz_loga.requires_grad = True
                layer.reset(1)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, loss_function, optimizer, lr_schedule, epoch,args.Lambda,layerID)
        state = {
            'importance': model.layers[layerID].qz_loga.data,
        }
        if not os.path.exists('Importance'):
            os.mkdir('Importance')
        torch.save(state, os.path.join('Importance','VGG19_layer_{}.pth.tar'.format(layerID)))

def Encrypt(model,Es,SEED):
    model.cpu()
    for layerID in Es.keys():
        E_locations = Es[layerID]
        model.layers[layerID].DPRM(E_locations, 'Gaussian',seed=SEED)
    return model

def validate(val_loader, model):
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'qz_loga'):
            layer.reset_to_10()
    top1 = AverageMeter()
    model.eval()
    model.cuda()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        # compute output
        output = model(input_var)
        output = output.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        top1.update(prec1, input.size(0))

    return top1.avg

def train(train_loader, model, criterion, optimizer, lr_schedule, epoch,Lambda,layerID):
    """Train for one epoch on the training set"""
    global total_steps, exp_flops, exp_l0, args, writer
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    lr_schedule.step(epoch=epoch)
    for i, (input_, target) in enumerate(train_loader):
        total_steps += 1
        if torch.cuda.is_available():
            target = target.cuda()
            input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        totalloss,loss, reg = criterion(model,output, target_var,layerID,Lambda)
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(totalloss.data, input_.size(0))
        top1.update(100 - prec1, input_.size(0))
        ## Adjust LR
        oldloss = totalloss
        if oldloss-totalloss > 1.0:
            optimizer.defaults['lr'] = optimizer.defaults['lr']*1
        # compute gradient and do SGD step
        optimizer.zero_grad()
        totalloss.backward()
        optimizer.step()
        # clamp the parameters
        layers = model.layers if not args.multi_gpu else model.module.layers
        for k, layer in enumerate(layers):
            if not isinstance(layer,nn.Linear):
                layer.constrain_parameters()
        TotalDataScale = len(train_loader.dataset)
        # input()
        IMPORTANCE = model.layers[layerID].qz_loga/(1+model.layers[layerID].qz_loga)
        MAX = torch.max(IMPORTANCE)
        MIN = torch.min(IMPORTANCE)
        if i == 0:
            Log = ('\nEpoch:[{0}][{1}/{2}], '
                  'Loss:{loss:.4f}, '
                  'Reg:{reg:.4f}, '
                  'Max Importance:{max:.4f}, ''Min Importance:{min:.4f}, '
                  'Lr:{lr:.4f}'.format(
                epoch, i, TotalDataScale,reg=reg, loss=loss, top1=top1,max=MAX,min=MIN,lr=optimizer.defaults['lr']))
        else:
            Log = ('\rEpoch:[{0}][{1}/{2}], '
                   'Loss:{loss:.4f}, '
                   'Reg:{reg:.4f}, '
                   'Max Importance:{max:.4f}, ''Min Importance:{min:.4f}, '
                   'Lr:{lr:.4f}'.format(
                epoch, i, len(train_loader), reg=reg, loss=loss, top1=top1, max=MAX, min=MIN,
                lr=optimizer.defaults['lr']))
        sys.stdout.write(Log)

    return top1.avg



if __name__ == '__main__':
    main()
