#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from models.resnet_simclr import ResNetSimCLR
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.test import test_img
from models.Fed import FedAvg
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Update import LocalUpdate
from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
import torch
from torchvision import datasets, transforms
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# FIX DATASET TO SPILT ACCORDING TO LABELS
# PLOT FOR DIFFERENT ITERATIONS AT EACH CLIENT


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    dataset = ContrastiveLearningDataset(args.data)

    dataset_train = dataset.get_dataset(args.dataset, args.n_views)
    dataset_test = dataset.get_test_dataset(args.dataset, args.n_views)

    # train_loader = torch.utils.data.DataLoader(
    #     dataset_train, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)

    # load dataset and split users
    if args.dataset == 'mnist':
        # trans_mnist = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # dataset_train = datasets.MNIST(
        #     '../data/mnist/', train=True, download=True, transform=trans_mnist)
        # dataset_test = datasets.MNIST(
        #     '../data/mnist/', train=False, download=True, transform=trans_mnist)

        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar10':
        # trans_cifar = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # dataset_train = datasets.CIFAR10(
        #     '../data/cifar', train=True, download=True, transform=trans_cifar)
        # dataset_test = datasets.CIFAR10(
        #     '../data/cifar', train=False, download=True, transform=trans_cifar)
        
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = len(dataset_train[0][0])

    # train_loader = torch.utils.data.DataLoader(
    #     dataset_train, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200,
                       dim_out=args.num_classes).to(args.device)
    elif args.model == 'resnet':
        net_glob = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    else:
        exit('Error: unrecognized model')
    
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    dict_glob = {}

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    epoch_loss = [[] for _ in range(args.num_users)]

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        dict_locals = [dict_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for i, idx in enumerate(idxs_users):
            local_model = LocalUpdate(
                args=args, index=idx, dataset=dataset_train, idxs=dict_users[idx])
            w, loss, epoch_losses = local_model.train(
                net=copy.deepcopy(net_glob).to(args.device))
            epoch_loss[i] += epoch_losses
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
                # dict_locals[idx] = copy.deepcopy(dict)
            else:
                w_locals.append(copy.deepcopy(w))
                # dict_locals.append(copy.deepcopy(dict))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)
        # for d in dict_locals:
        #     dict_glob.update(d)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        # MAKE WEIGHTED AVERAGE
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        for i in range(args.local_ep):
            loss_train.append(loss_avg)

    # Plot for clients
    for index in range(args.num_users):
        plt.figure()
        plt.plot(range(len(epoch_loss[index])), epoch_loss[index])
        plt.ylabel('train_loss')
        plt.savefig('./save/client{}_{}_{}_{}_C{}_iid{}.png'.format(index, args.dataset,
                                                                    args.model, args.epochs, args.frac, args.iid))


    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset,
                                                           args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    dataset_train = datasets.CIFAR10(args.data, train=True, download=True,
                                     transform=transforms.ToTensor())
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
