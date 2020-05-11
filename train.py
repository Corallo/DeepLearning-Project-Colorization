from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np

import loss
from model import *
import utils
import argparse
import os
import time
from tensorboardX import SummaryWriter
from datasets import ImageNet


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
parser.add_argument('--run_dir', default='', type=str)
parser.add_argument('--kmeans_source', default='imagenet/', type=str)



def weights_init(m, args):
    #kmeans_init.kmeans_init(m, utils.load_images(args),num_iter=3, use_whitening=False)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.init.constant_(m.bias.data, 0.1)

best_prec1 = 0
writer = SummaryWriter()

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    if args.run_dir == '':
        writer = SummaryWriter()
    else:
        print("=> Logs can be found in", args.run_dir)
        writer = SummaryWriter(args.run_dir)

    # create model
    print("=> creating model")


    model = nn.DataParallel(NNet()).cuda()

    # print("paralleling")
    # model = torch.nn.DataParallel(model, device_ids=range(args.nGpus)).cuda()
    weights_init(model,args)
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

    # define loss function (criterion) and optimizer
    criterion = loss.classificationLoss
    optimizer = torch.optim.Adam([{'params': model.parameters()},], args.lr,weight_decay=args.weight_decay, betas=(0.9, 0.99))


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        print("=> Epoch", epoch, "started.")
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        }, args.reduced)
        print("=> Epoch", epoch, "finished.")


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
        if torch.isnan(loss):
            print('NaN value encountered in loss.')
            continue
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
        if (i+1) % 1000 == 0:
            start = time.time()
            batch_num = np.maximum(args.batch_size//4,2)
            idx = i + epoch*len(train_loader)
            imgs = utils.getImages(img, target, output.detach().cpu(), batch_num)
            writer.add_image('data/imgs_gen', imgs, idx)
            print("Img conversion time: ", time.time() - start)
        writer.add_scalar('data/loss_train', losses.avg, i + epoch*len(train_loader))


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
    lr = 3e-5
    if epoch >= 2:
        lr = 1e-5
    if epoch >= 5:
        lr = 3e-6

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """Computes the accuracy for image k"""

    return acc


if __name__ == '__main__':
    main()
