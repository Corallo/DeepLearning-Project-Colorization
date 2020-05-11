from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.nn.functional import interpolate
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


    model_G = nn.DataParallel(NNet()).cuda()
    model_D = nn.DataParallel(DCGAN()).cuda()

    weights_init(model_G, args)
    weights_init(model_D, args)
    print("=> model weights initialized")
    print(model_G)
    print(model_D)

    # optionally resume from a checkpoint
    if args.resume: 
        for (path, net_G, net_D) in [(args.resume, model_G, model_D)]:
            if os.path.isfile(path):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(path)
                args.start_epoch = checkpoint['epoch']

                net_G.load_state_dict(checkpoint['state_dict_G'])
                net_D.load_state_dict(checkpoint['state_dict_D'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(path))

    # Data loading code
    train_root = args.train_root
    train_dataset = ImageNet(train_root, output_full=True)

    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=8, pin_memory=True)
        print("=> Loaded data, length = ", len(train_dataset))

    # define loss function (criterion) and optimizer
    criterion_G = nn.MSELoss()
    criterion_GAN = nn.BCEWithLogitsLoss()
    def GANLoss(pred, is_real):
        if is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        return criterion_GAN(pred, target)
    optimizer_G = torch.optim.Adam([{'params': model_G.parameters()},], args.lr,weight_decay=args.weight_decay, betas=(0.9, 0.99))
    optimizer_D = torch.optim.Adam([{'params': model_D.parameters()},], args.lr,weight_decay=args.weight_decay, betas=(0.9, 0.999))

    for epoch in range(args.start_epoch, args.epochs):
        print("=> Epoch", epoch, "started.")
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model_G, model_D, criterion, optimizer, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_G': model_G.state_dict(),
            'state_dict_D': model_D.state_dict(),
        }, args.reduced)
        print("=> Epoch", epoch, "finished.")


def train(train_loader, model_G, model_D, criterion, optimizer, epoch):

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_L2 = AverageMeter()
    losses_G = AverageMeter()
    losses_D = AverageMeter()

    end = time.time()
    for i, (real, img_L, target) in enumerate(train_loader):

        ## Code for forward - backward - update pass in Generator and Discriminator
        # Inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

        data_time.update(time.time() - end)
        var = Variable(img_L.float(), requires_grad=True).cuda()
        real = Variable(real.float(), requires_grad=True).cuda()
        # compute output G(L)
        output = model_G(var)

        # Update gradients for Discriminator
        model_D.set_grads(True)
        optimizer_D.zero_grad()
        # Fake loss term
        output_up = interpolate(output, scale_factor=4, mode='bilinear', 
            recompute_scale_factor=True, align_corners=True)
        fake_img = torch.cat([img_L, output_up], 1)
        fake_prob = model_D(fake_img.detach())
        loss_D_fake = GANLoss(fake_prob, False)
        # Real loss term
        real_prob = model_D(real)
        loss_D_real = GANLoss(real_prob, True)

        loss_D = (loss_D_real + loss_D_fake)*0.5
        if torch.isnan(loss_D):
            print('NaN value encountered in loss_D.')
            continue
        loss_D.backward()
        optimizer_D.step()

        # Update gradients for Generator
        model_D.set_grads(False)
        optimizer_G.zero_grad()
        fake_prob = model_D(fake_img)
        # Fool the discriminator
        loss_G_GAN = GANLoss(fake_prob, True)
        # Regressor loss term
        loss_G_L2 = criterion_G(output, target)
        loss_G = loss_G_GAN + loss_G_L2*10
        loss_G.backward()
        optimizer_G.step()

        losses_D.update(loss_D.data, var.size(0))
        losses_G.update(loss_G_GAN.data, var.size(0))
        losses_L2.update(loss_G_L2.data, var.size(0))

        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: G {loss_g.val:.4f} ({loss_g.avg:.4f})\t'
                  'D {loss_d.val:.4f} ({loss_d.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_g=losses_G, loss_d=losses_D))
        if (i+1) % 5000 == 0:
            print("Saving checkpoint...")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict_G': model_G.state_dict(),
                'state_dict_D': model_D.state_dict(),
            }, args.reduced)
        if (i+1) % 1000 == 0:
            start = time.time()
            batch_num = np.maximum(args.batch_size//4,2)
            idx = i + epoch*len(train_loader)
            imgs = utils.getImages(img, target, output.detach().cpu(), batch_num, decode=False)
            writer.add_image('data/imgs_gen', imgs, idx)
            print("Img conversion time: ", time.time() - start)
        writer.add_scalar('data/L2_loss_train', losses_L2.avg, i + epoch*len(train_loader))
        writer.add_scalar('data/GAN_loss_train', {
            'loss_G': losses_G.avg,
            'loss_D': losses_D.avg
        }, i + epoch*len(train_loader))


def save_checkpoint(state, reduced, filename='model'):
    if reduced:
        torch.save(state, "models/" + filename + '_reduced_latest.pth.tar')
    else:
        torch.save(state, "models/" + filename + '_gan_latest.pth.tar')


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

    for param_group in optimizer_G.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_D.param_groups:
        param_group['lr'] = lr*10


def accuracy(output, target):
    """Computes the accuracy for image k"""

    return acc


if __name__ == '__main__':
    main()
