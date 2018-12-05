import numpy as np

import pickle
from PIL import Image
import multiprocessing
import time
import os
import tqdm
import shutil
from random import randint
import argparse
import resnet

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils
from model.network import resnet101
import dataloader.motion_dataloader as motion_dataloader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ROOT = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
    description='UCF101 motion stream on resnet101')
parser.add_argument(
    '--epochs',
    default=20,
    type=int,
    metavar='N',
    help='number of total epochs')
parser.add_argument(
    '--batch-size',
    default=8,
    type=int,
    metavar='N',
    help='mini-batch size (default: 16)')
parser.add_argument(
    '--lr',
    default=1e-2,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--model',
    default="ImageNet",
    dest='model',
    help='type of pretraining to use (No, ImageNet, Transfer)')
parser.add_argument(
    '--transfer-path',
    dest='transfer_path',
    default='/vision/vision_users/azou/data/models/fusionseg_binary_20000_1e-5.pth',
    help='path to transfer model (only used if --model=Transfer)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '--dataset',
    dest='dataset',
    default="UCF101",
    help='dataset to use (UCF101 or HMDB51)')
parser.add_argument(
    '--output-dir',
    dest='output_dir',
    default=os.path.join(ROOT, 'record/motion'),
    help="directory to save state to")
parser.add_argument(
    '--flow-dir',
    dest='flow_dir',
    default='/vision/vision_users/azou/data/ucf101_flow/',
    help="root directory of flow data")
parser.add_argument(
    '--split-dir',
    dest='split_dir',
    default='/vision/vision_users/azou/data/ecf101_splits/',
    help="directory containing train/test split files")


def main():
    global arg
    arg = parser.parse_args()
    print(arg)

    # Prepare DataLoader
    data_loader = motion_dataloader.Motion_DataLoader(
        BATCH_SIZE=arg.batch_size,
        num_workers=multiprocessing.cpu_count(),
        path=arg.flow_dir,
        list_path=arg.split_dir,
        split='01',
        in_channel=10,
        dataset_type=motion_dataloader.DataSetType[arg.dataset])

    train_loader, test_loader, test_video = data_loader.run()

    # Bookkeeping
    try:
        os.makedirs(arg.output_dir)
    except OSError:
        print('Directories already exist')
    if arg.dataset == "UCF101":
        num_classes = 101
        zero_indexed = False
    elif arg.dataset == "HMDB51":
        num_classes = 51
        zero_indexed = True
    else:
        raise ValueError("Only UCF101 and HMDB51 are supported")

    # Model
    model = Motion_CNN(
        model_type=arg.model,
        transfer_path=arg.transfer_path,
        # Data Loader
        train_loader=train_loader,
        test_loader=test_loader,
        # Utility
        start_epoch=arg.start_epoch,
        resume=arg.resume,
        evaluate=arg.evaluate,
        output_dir=arg.output_dir,
        zero_indexed=zero_indexed,
        num_classes=num_classes,
        # Hyper-parameter
        nb_epochs=arg.epochs,
        lr=arg.lr,
        batch_size=arg.batch_size,
        channel=10 * 2,
        test_video=test_video)
    # Training
    model.run()


class Motion_CNN():
    def __init__(self, model_type, transfer_path, nb_epochs, lr, batch_size,
                 resume, start_epoch, evaluate, output_dir, num_classes,
                 zero_indexed, train_loader, test_loader, channel, test_video):
        self.model_type = model_type
        self.transfer_path = transfer_path
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.resume = resume
        self.zero_indexed = zero_indexed
        self.start_epoch = start_epoch
        self.evaluate = evaluate
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_prec1 = 0
        self.channel = channel
        self.test_video = test_video

    def build_model(self):
        print('==> Build model and setup loss and optimizer')
        # build model
        if self.model_type.lower() == "no":
            self.model = resnet101(
                pretrained=False,
                channel=self.channel,
                num_classes=self.num_classes).cuda()
        elif self.model_type.lower() == "ImageNet".lower():
            self.model = resnet101(
                pretrained=True,
                channel=self.channel,
                num_classes=self.num_classes).cuda()
        elif self.model_type.lower() == "Transfer".lower():
            # Pretraining option for transfer
            model = resnet.getResNetFromDeepLabV2(self.transfer_path,
                                                  self.num_classes)
            self.model = resnet101(
                pretrained=True,
                channel=self.channel,
                num_classes=self.num_classes,
                dict=model.state_dict()).cuda()

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         self.lr,
                                         momentum=0.9)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', patience=1, verbose=True)

    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print(
                    "==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                    .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True

        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            is_best = prec1 > self.best_prec1
            # lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_prec1 = prec1
                with open(
                        os.path.join(self.output_dir,
                                     'motion_video_preds_{}.pickle'.format(
                                         self.model_type)), 'wb') as f:
                    pickle.dump(self.dic_video_level_preds, f)
                f.close()

            utils.save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer': self.optimizer.state_dict()
            }, is_best,
                os.path.join(self.output_dir, 'checkpoint.pth'),
                os.path.join(self.output_dir, 'model_best.pth'))

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch,
                                                           self.nb_epochs))
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        # switch to train mode
        self.model.train()
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data, label) in enumerate(progress):

            # measure data loading time
            data_time.update(time.time() - end)

            label = label.cuda(async=True)
            input_var = Variable(data).cuda()
            target_var = Variable(label).cuda()

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = utils.accuracy(output.data, label, topk=(1, 5))
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        info = {
            'Epoch': [self.epoch],
            'Batch Time': [round(batch_time.avg, 3)],
            'Data Time': [round(data_time.avg, 3)],
            'Loss': [round(losses.avg, 5)],
            'Prec@1': [round(top1.avg, 4)],
            'Prec@5': [round(top5.avg, 4)],
            'lr': self.optimizer.param_groups[0]['lr']
        }
        utils.record_info(info,
                          os.path.join(self.output_dir,
                                       'opf_train_{}.csv'.format(self.model_type)),
                          'train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch,
                                                             self.nb_epochs))

        batch_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds = {}
        end = time.time()
        progress = tqdm(self.test_loader)
        for i, (keys, data, label) in enumerate(progress):
            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]
            for j in range(nb_data):
                videoName = keys[j].split('|', 1)[0]  # ApplyMakeup_g01_c01
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j, :]
                else:
                    self.dic_video_level_preds[videoName] += preds[j, :]

        # Frame to video level accuracy
        video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()
        info = {
            'Epoch': [self.epoch],
            'Batch Time': [np.round(batch_time.avg, 3)],
            'Loss': [np.round(video_loss, 5)],
            'Prec@1': [np.round(video_top1, 3)],
            'Prec@5': [np.round(video_top5, 3)]
        }
        record_info(info,
                    os.path.join(self.output_dir,
                                 'opf_test_{}.csv'.format(self.model_type)),
                    'test')
        return video_top1, video_loss

    def frame2_video_level_accuracy(self):
        correct = 0
        video_level_preds = np.zeros(
            (len(self.dic_video_level_preds), self.num_classes))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii = 0
        for key in sorted(self.dic_video_level_preds.keys()):
            name = key.split('|', 1)[0]

            preds = self.dic_video_level_preds[name]
            if not self.zero_indexed:
                label = int(self.test_video[name]) - 1
            else:
                label = int(self.test_video[name])

            video_level_preds[ii, :] = preds
            video_level_labels[ii] = label
            ii += 1
            if np.argmax(preds) == (label):
                correct += 1

        # top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()

        loss = self.criterion(
            Variable(video_level_preds).cuda(),
            Variable(video_level_labels).cuda())
        top1, top5 = utils.accuracy(
            video_level_preds, video_level_labels, topk=(1, 5))

        top1 = float(top1.numpy())
        top5 = float(top5.numpy())

        return top1, top5, loss.data.cpu().numpy()


if __name__ == '__main__':
    main()
