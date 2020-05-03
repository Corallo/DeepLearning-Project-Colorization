
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from model import NNet

import argparse
import os
import shutil
import time
from tensorboardX import SummaryWriter




parser = argparse.ArgumentParser(description='PyTorch Incidents Training')

parser.add_argument('--train_root', default='/dataset/train', type=str)
parser.add_argument('--val_root', default='/dataset/val', type=str)
parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--nGpus', default=2, type=int, help='number of gpus to use')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='models/model.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',help='use pre-trained model')

best_prec1 = 0
writer = SummaryWriter()

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    # create model
    print("=> creating model")

    model = NNet()

    # print("paralleling")
    # model = torch.nn.DataParallel(model, device_ids=range(args.nGpus)).cuda()
        
    print(model)

    # optionally resume from a checkpoint
    if args.resume: 
        for (path, net) in [(args.resume, model)]:
            if os.path.isfile(path):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(path)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                net.load_state_dict(checkpoint['state_dict'])

                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(path))

    # Ni idea de que es
    cudnn.benchmark = True

    # Data loading code
    train_root = args.train_root
    val_root = args.val_root
    train_dataset = datasets.ImageFolder(train_root)
    val_dataset = datasets.ImageFolder(val_root)
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD([{'params': model.parameters()},], args.lr,momentum=args.momentum,weight_decay=args.weight_decay)


    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1 
        best_prec1 = max(prec1, best_prec1)
        for prefix in prefix2model:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prefix2model[prefix].state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (img, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        var = Variable(img, requires_grad=True).unsqueeze(0).unsqueeze(0)
        # compute output
        output = model(var)
        # record loss
        loss = criterion()
        # measure accuracy and record loss
        prec1, = accuracy(output.data, target)
        losses.update(loss.data, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        writer.add_scalar('data/loss_train', losses.avg, i + epoch*len(train_loader))


def validate(val_loader, model, criterion, epoch):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (img, target) in enumerate(val_loader):
        var = Variable(img, requires_grad=True).unsqueeze(0).unsqueeze(0)
        # compute output
        output = model(var)
        loss = cirterion()

        # measure accuracy and record loss
        prec1,  = accuracy(output.data, target)
        losses.update(loss.data, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))
    writer.add_scalar('data/loss_val', losses.avg, epoch)


    return 


def save_checkpoint(state, is_best, filename='model'):
    torch.save(state, "models/" + filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile("models/" + filename + '_latest.pth.tar', "models/" + filename + '_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every n epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """Computes the accuracy for image k"""

    return acc


if __name__ == '__main__':
    main()
