from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms


import loss
from model import *
import utils
from k_means_init import kmeans_init
import argparse
import os
import time
from tensorboardX import SummaryWriter
from datasets import ImageNet

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)

parser = argparse.ArgumentParser(description='PyTorch Incidents Training')

parser.add_argument('--train_root', default='/dataset/train', type=str)
parser.add_argument('--val_root', default='/dataset/val', type=str)
parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--nGpus', default=1, type=int, help='number of gpus to use')
parser.add_argument('--lr', '--learning-rate', default=3e-5, type=float,metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='models/model.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',help='use pre-trained model')
parser.add_argument('--reduced', dest='reduced', action='store_true', help='use reduced-model')

best_prec1 = 0
writer = SummaryWriter()

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    # create model
    print("=> creating model")

    if args.reduced:
        model = nn.DataParallel(NNetReduced()).cuda()
    else:
        model = nn.DataParallel(NNet()).cuda()

    # print("paralleling")
    # model = torch.nn.DataParallel(model, device_ids=range(args.nGpus)).cuda()
    model.apply(weights_init)
    print("=> model weights initialized")
    print(model)

    # optionally resume from a checkpoint
    if args.resume: 
        for (path, net) in [(args.resume, model)]:
            if os.path.isfile(path):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(path)
                args.start_epoch = checkpoint['epoch']
                #best_prec1 = checkpoint['best_prec1']
                net.load_state_dict(checkpoint['state_dict'])

                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(path))

    # Data loading code
    train_root = args.train_root
    #val_root = args.val_root
    train_dataset = ImageNet(train_root)
    #val_dataset = datasets.ImageFolder(val_root)
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=8, pin_memory=True)
        print("=> Loaded data, length = ", len(train_dataset))
    #val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True)

    #input_image, _ = next(iter(train_loader))
    #model = kmeans_init(model, input_image.double().cuda(), 3, True)


    # define loss function (criterion) and optimizer
    criterion = loss.classificationLoss
    optimizer = torch.optim.Adam([{'params': model.parameters()},], args.lr,weight_decay=args.weight_decay, betas=(0.9, 0.99))


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        #prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        #is_best = prec1 > best_prec1 
        #best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, args.reduced)


def train(train_loader, model, criterion, optimizer, epoch):

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (img, target) in enumerate(train_loader):
        # measure data loading time
        if img is None:
            continue
        data_time.update(time.time() - end)
        encoded_target = Variable(utils.soft_encode_ab(target).float(), requires_grad=False).cuda()
        var = Variable(img.float(), requires_grad=True).cuda()
        # compute output
        output = model(var)
        # record loss
        loss = criterion(output, encoded_target)
        # measure accuracy and record loss
        #prec1, = accuracy(output.data, target)
        losses.update(loss.data, var.size(0))

        # compute gradient and do SGD step
        backwardTime = time.time()
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
        if (i+1) % 5000 == 0:
            print("Saving checkpoint...")
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }, args.reduced)
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


def save_checkpoint(state, reduced, filename='model'):
    if reduced:
        torch.save(state, "models/" + filename + '_reduced_latest.pth.tar')
    else:
        torch.save(state, "models/" + filename + '_latest.pth.tar')


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
