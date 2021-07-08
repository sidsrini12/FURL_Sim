#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import logging
import os
import sys

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, index, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss().to(self.args.device)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.writer = SummaryWriter()
        self.index = index

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(self.ldr_train), eta_min=0,
                                                               last_epoch=-1)

        scaler = GradScaler(enabled=self.args.fp16_precision)
        epoch_loss = []
        n_iter = 0
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                logits, labels = self.info_nce_loss(log_probs)
                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                # if n_iter % self.args.log_every_n_steps == 0:
                #     top1, top5 = accuracy(logits, labels, topk=(1, 5))
                #     self.writer.add_scalar('loss', loss, global_step=n_iter)
                #     self.writer.add_scalar(
                #         'acc/top1', top1[0], global_step=n_iter)
                #     self.writer.add_scalar(
                #         'acc/top5', top5[0], global_step=n_iter)
                #     self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[
                #                            0], global_step=n_iter)

                n_iter += 1
                batch_loss.append(loss.item())

            # warmup for the first 10 epochs
            if iter >= 10:
                self.scheduler.step()
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        # plot loss curve
        # plt.figure()
        # plt.plot(range(len(epoch_loss)), epoch_loss)
        # plt.ylabel('train_loss')
        # plt.savefig('./save/client{}_{}_{}_{}_C{}_iid{}.png'.format(self.index, self.args.dataset,
        #                                                     self.args.model, self.args.epochs, self.args.frac, self.args.iid))
        top1, top5 = accuracy(logits, labels, topk=(1, 5))
        print("Accuracy: Top1 {} Top5 {}".format(top1[0], top5[0]))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), epoch_loss

        
    # def __init__(self, *args, **kwargs):
    #     self.args = kwargs['args']
    #     self.model = kwargs['model'].to(self.args.device)
    #     self.optimizer = kwargs['optimizer']
    #     self.scheduler = kwargs['scheduler']
    #     self.writer = SummaryWriter()
    #     logging.basicConfig(filename=os.path.join(
    #         self.writer.log_dir, 'training.log'), level=logging.DEBUG)
    #     self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.local_bs)
                            for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(
            self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(
            logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    # def train(self, train_loader):

    #     scaler = GradScaler(enabled=self.args.fp16_precision)

    #     # save config file
    #     save_config_file(self.writer.log_dir, self.args)

    #     n_iter = 0
    #     logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
    #     logging.info(f"Training with gpu: {self.args.disable_cuda}.")

    #     for epoch_counter in range(self.args.epochs):
    #         for images, _ in tqdm(train_loader):
    #             images = torch.cat(images, dim=0)

    #             images = images.to(self.args.device)

    #             with autocast(enabled=self.args.fp16_precision):
    #                 features = self.model(images)
    #                 logits, labels = self.info_nce_loss(features)
    #                 loss = self.criterion(logits, labels)

    #             self.optimizer.zero_grad()

    #             scaler.scale(loss).backward()

    #             scaler.step(self.optimizer)
    #             scaler.update()

    #             if n_iter % self.args.log_every_n_steps == 0:
    #                 top1, top5 = accuracy(logits, labels, topk=(1, 5))
    #                 self.writer.add_scalar('loss', loss, global_step=n_iter)
    #                 self.writer.add_scalar(
    #                     'acc/top1', top1[0], global_step=n_iter)
    #                 self.writer.add_scalar(
    #                     'acc/top5', top5[0], global_step=n_iter)
    #                 self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[
    #                                        0], global_step=n_iter)

    #             n_iter += 1

    #         # warmup for the first 10 epochs
    #         if epoch_counter >= 10:
    #             self.scheduler.step()
    #         logging.debug(
    #             f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

    #     logging.info("Training has finished.")
    #     # save model checkpoints
    #     checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
    #     save_checkpoint({
    #         'epoch': self.args.epochs,
    #         'arch': self.args.arch,
    #         'state_dict': self.model.state_dict(),
    #         'optimizer': self.optimizer.state_dict(),
    #     }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
    #     logging.info(
    #         f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
