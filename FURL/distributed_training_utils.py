from typing import DefaultDict
import torch, torchvision
import numpy as np
import itertools as it
import re
from math import sqrt
import random
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import time

import compression_utils as comp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def copy(target, source):
  for name in target:
    target[name].data = source[name].data.clone()

def add(target, source):
  for name in target:
    target[name].data += source[name].data.clone()

def scale(target, scaling):
  for name in target:
    target[name].data = scaling*target[name].data.clone()

def subtract(target, source):
  for name in target:
    target[name].data -= source[name].data.clone()

def subtract_(target, minuend, subtrahend):
  for name in target:
    target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()

def average(target, sources):
  for name in target:
    target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()

def weighted_average(target, sources, weights):
  for name in target:
    summ = torch.sum(weights)
    n = len(sources)
    modify = [weight/summ*n for weight in weights]
    target[name].data = torch.mean(torch.stack([m*source[name].data for source, m in zip(sources, modify)]), dim=0).clone()

def majority_vote(target, sources, lr):
  for name in target:
    threshs = torch.stack([torch.max(source[name].data) for source in sources])
    mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
    target[name].data = (lr*mask).clone()

def compress(target, source, compress_fun):
  '''
  compress_fun : a function f : tensor (shape) -> tensor (shape)
  '''
  for name in target:
    target[name].data = compress_fun(source[name].data.clone())


def fedCA(x, y, z_dict, d_align, W_align):
  # Obtain x_i and x_j from x
  x_i = x[1:5]
  x_j = x[5:]

  z_i = normalized_projections(x_i)
  z_j = normalized_projections(x_j)

  logits_batch = np.dot(z_i, z_j.T)
  logits_dict = np.dot(z_i, z_dict.T)

  logits_total = dict_concatenate([logits_batch, logits_dict], 1)  # logits is N x (N+K)

  N = len(x)
  labels = list(range(N))

  loss_contrastive = nn.CrossEntropyLoss()
  loss_cont = loss_contrastive(logits_total, labels) # division by temperature??

  # Find loss_align

  beta = 0.5 # Scaling factor for loss_align
  loss = loss_contrastive + beta * loss_align

      
class DistributedTrainingDevice(object):
  '''
  A distributed training device (Client or Server)
  data : a pytorch dataset consisting datapoints (x,y)
  model : a pytorch neural net f mapping x -> f(x)=y_
  hyperparameters : a python dict containing all hyperparameters
  '''
  def __init__(self, dataloader, model, hyperparameters, experiment):
    self.hp = hyperparameters
    self.xp = experiment
    self.loader = dataloader
    self.model = model
    self.loss_fn = nn.CrossEntropyLoss()


class Client(DistributedTrainingDevice):

  def __init__(self, dataloader, model, hyperparameters, experiment, id_num=0):
    super().__init__(dataloader, model, hyperparameters, experiment)

    self.id = id_num

    # Parameters
    self.W = {name : value for name, value in self.model.named_parameters()}
    self.W_old = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    self.dW = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    self.dW_compressed = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    self.A = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}

    self.n_params = sum([T.numel() for T in self.W.values()])
    self.bits_sent = []

    self.dict = defaultdict()

    # Optimizer (specified in self.hp, initialized using the suitable parameters from self.hp)
    optimizer_object = getattr(optim, self.hp['optimizer'])
    optimizer_parameters = {k : v for k, v in self.hp.items() if k in optimizer_object.__init__.__code__.co_varnames}

    self.optimizer = optimizer_object(self.model.parameters(), **optimizer_parameters)

    # Learning Rate Schedule
    self.scheduler = getattr(optim.lr_scheduler, self.hp['lr_decay'][0])(self.optimizer, **self.hp['lr_decay'][1])

    # State
    self.epoch = 0
    self.train_loss = 0.0


  def synchronize_with_server(self, server):
    # W_client = W_server
    copy(target=self.W, source=server.W)


  def train_cnn(self, iterations):

    running_loss = 0.0
    for i in range(iterations):
      
      try: # Load new batch of data
        x, y = next(self.epoch_loader)
      except: # Next epoch
        self.epoch_loader = iter(self.loader)
        self.epoch += 1

        # Adapt lr according to schedule
        if isinstance(self.scheduler, optim.lr_scheduler.LambdaLR):
          self.scheduler.step()
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) and 'loss_test' in self.xp.results:
          self.scheduler.step(self.xp.results['loss_test'][-1])
        
        x, y = next(self.epoch_loader)

      x, y = x.to(device), y.to(device)
        
      # zero the parameter gradients
      self.optimizer.zero_grad()

      # forward + backward + optimize
      y_ = self.model(x)

      loss = self.loss_fn(y_, y)
      loss.backward()
      self.optimizer.step()
      
      running_loss += loss.item()

    return running_loss / iterations


  def compute_weight_update(self, iterations=1):

    # Training mode
    self.model.train()

    # W_old = W
    copy(target=self.W_old, source=self.W)
    
    # W = SGD(W, D)
    self.train_loss = self.train_cnn(iterations)

    # dW = W - W_old
    subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

  
  def compress_weight_update_up(self, compression=None, accumulate=False, count_bits=False):

    if accumulate and compression[0] != "none":
      # compression with error accumulation     
      add(target=self.A, source=self.dW)
      compress(target=self.dW_compressed, source=self.A, compress_fun=comp.compression_function(*compression))
      subtract(target=self.A, source=self.dW_compressed)

    else: 
      # compression without error accumulation
      compress(target=self.dW_compressed, source=self.dW, compress_fun=comp.compression_function(*compression))

    if count_bits:
      # Compute the update size
      self.bits_sent += [comp.get_update_size(self.dW_compressed, compression)]

  def clientUpdate(self, dict, iterations=5):
    for i in range(self.epoch):
      for j in range(iterations):
        x,y = next(iter(self.loader))
        # Training mode
        self.model.train()

        # W_old = W
        copy(target=self.W_old, source=self.W)

        # W = SGD(W, D)
        self.train_loss = self.fedCA(x,y,dict,d_align, W_align)

        # dW = W - W_old
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

    # Assemble dictionary

    return self.W, self.dict

  def doClusteringWithKMeansLoss(self, dataset, epochs):
    '''
      Trains the autoencoder with combined kMeans loss and reconstruction loss
      At the moment does not give good results
      :param dataset: Data on which the autoencoder is trained
      :param epochs: Number of training epochs
      :return: None - (side effect) saves the trained network params and latent space in appropriate location
    '''
    batch_size = self.batch_size
    # Load the inputs in latent space produced by the pretrained autoencoder and use it to initialize cluster centers
    Z = np.load('saved_params/%s/z_%s.npy' % (dataset.name, self.name))
    quality_desc, cluster_centers = evaluateKMeans(
        Z, dataset.labels, dataset.getClusterCount(), 'Initial')
    rootLogger.info(quality_desc)
    # Load network parameters - code borrowed from mnist lasagne example
    with np.load('saved_params/%s/m_%s.npz' % (dataset.name, self.name)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(
            self.network, param_values, trainable=True)
    # reconstruction loss is just rms loss between input and reconstructed input
    reconstruction_loss = self.getReconstructionLossExpression(
        layers.get_output(self.network), self.t_target)
    # extent the network to do soft cluster assignments
    clustering_network = ClusteringLayer(self.encode_layer, dataset.getClusterCount(
    ), cluster_centers, batch_size, self.encode_size)
    soft_assignments = layers.get_output(clustering_network)
    # k-means loss is the sum of distances from the cluster centers weighted by the soft assignments to the clusters
    kmeansLoss = self.getKMeansLoss(layers.get_output(self.encode_layer), soft_assignments,
                                    clustering_network.W, dataset.getClusterCount(), self.encode_size, batch_size)
    params = lasagne.layers.get_all_params(self.network, trainable=True)
    # total loss = reconstruction loss + lambda * kmeans loss
    weight_reconstruction = 1
    weight_kmeans = 0.1
    total_loss = weight_kmeans * kmeansLoss + \
        weight_reconstruction * reconstruction_loss
    updates = lasagne.updates.nesterov_momentum(
        total_loss, params, learning_rate=0.01)
    trainKMeansWithAE = theano.function(
        [self.t_input, self.t_target], total_loss, updates=updates)
    for epoch in range(epochs):
        error = 0
        total_batches = 0
        for batch in dataset.iterate_minibatches(self.input_type, batch_size, shuffle=True):
            inputs, targets = batch
            error += trainKMeansWithAE(inputs, targets)
            total_batches += 1
        # For every 10th epoch, update the cluster centers and print the clustering accuracy and nmi - for checking if the network
        # is actually doing something meaningful - the labels are never used for training
        if (epoch + 1) % 10 == 0:
            for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
                Z[i * batch_size:(i + 1) *
                  batch_size] = self.predictEncoding(batch[0])
            quality_desc, cluster_centers = evaluateKMeans(Z, dataset.labels, dataset.getClusterCount(
            ), "%d/%d [%.4f]" % (epoch + 1, epochs, error / total_batches))
            rootLogger.info(quality_desc)
        else:
            # Just print the training loss
            rootLogger.info("%-30s     %8s     %8s" %
                            ("%d/%d [%.4f]" % (epoch + 1, epochs, error / total_batches), "", ""))
        if self.shouldStopNow:
          break

    # Save the inputs in latent space and the network parameters
    for i, batch in enumerate(dataset.iterate_minibatches(self.input_type, batch_size, shuffle=False)):
        Z[i * batch_size:(i + 1) *
          batch_size] = self.predictEncoding(batch[0])
    np.save('saved_params/%s/pc_km_z_%s.npy' %
            (dataset.name, self.name), Z)
    np.savez('saved_params/%s/pc_km_m_%s.npz' % (dataset.name, self.name),
              *lasagne.layers.get_all_param_values(self.network, trainable=True))

  def getKMeansLoss(self, latent_space_expression, soft_assignments, t_cluster_centers, num_clusters, latent_space_dim, num_samples, soft_loss=False):
    # Kmeans loss = weighted sum of latent space representation of inputs from the cluster centers
    z = latent_space_expression.reshape((num_samples, 1, latent_space_dim))
    z = T.tile(z, (1, num_clusters, 1))
    u = t_cluster_centers.reshape((1, num_clusters, latent_space_dim))
    u = T.tile(u, (num_samples, 1, 1))
    distances = (z - u).norm(2, axis=2).reshape((num_samples, num_clusters))
    if soft_loss:
        weighted_distances = distances * soft_assignments
        loss = weighted_distances.sum(axis=1).mean()
    else:
        loss = distances.min(axis=1).mean()
    return loss



      



class Server(DistributedTrainingDevice):

  def __init__(self, dataloader, model, hyperparameters, experiment, stats):
    super().__init__(dataloader, model, hyperparameters, experiment)

    # Prepare d_align and W_align public dataset

    # Parameters
    self.W = {name : value for name, value in self.model.named_parameters()}
    self.dW_compressed = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    self.dW = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}

    self.A = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}

    self.n_params = sum([T.numel() for T in self.W.values()])
    self.bits_sent = []

    self.client_sizes = torch.Tensor(stats["split"]).cuda()



  def aggregate_weight_updates(self, clients, aggregation="mean"):

    # dW = aggregate(dW_i, i=1,..,n)
    if aggregation == "mean":
      average(target=self.dW, sources=[client.dW_compressed for client in clients])

    elif aggregation == "weighted_mean":
      weighted_average(target=self.dW, sources=[client.dW_compressed for client in clients], 
        weights=torch.stack([self.client_sizes[client.id] for client in clients]))
    
    elif aggregation == "majority":
      majority_vote(target=self.dW, sources=[client.dW_compressed for client in clients], lr=self.hp["lr"])


  def compress_weight_update_down(self, compression=None, accumulate=False, count_bits=False):
    if accumulate and compression[0] != "none":
      # compression with error accumulation   
      add(target=self.A, source=self.dW)
      compress(target=self.dW_compressed, source=self.A, compress_fun=comp.compression_function(*compression))
      subtract(target=self.A, source=self.dW_compressed)

    else: 
      # compression without error accumulation
      compress(target=self.dW_compressed, source=self.dW, compress_fun=comp.compression_function(*compression))

    add(target=self.W, source=self.dW_compressed)

    if count_bits:
      # Compute the update size
      self.bits_sent += [comp.get_update_size(self.dW_compressed, compression)]

 
  def evaluate(self, loader=None, max_samples=50000, verbose=True):
    """Evaluates local model stored in self.W on local dataset or other 'loader if specified for a maximum of 
    'max_samples and returns a dict containing all evaluation metrics"""
    self.model.eval()

    eval_loss, correct, samples, iters = 0.0, 0, 0, 0
    if not loader:
      loader = self.loader
    with torch.no_grad():
      for i, (x,y) in enumerate(loader):

        x, y = x.to(device), y.to(device)
        y_ = self.model(x)
        _, predicted = torch.max(y_.data, 1)
        eval_loss += self.loss_fn(y_, y).item()
        correct += (predicted == y).sum().item()
        samples += y_.shape[0]
        iters += 1

        if samples >= max_samples:
          break
      if verbose:
        print("Evaluated on {} samples ({} batches)".format(samples, iters))
  
      results_dict = {'loss' : eval_loss/iters, 'accuracy' : correct/samples}

    return results_dict

  def execute(self, rounds, C, num_clients, batch_size, clients):
    for i in range(rounds):
      m = max(C*num_clients, 1)

      batch_ind = np.random.choice(num_clients, batch_size)

      for u in clients[batch_ind]:
        u.W, u.dict = u.clientUpdate()

      self.W = weighted_average(clients[batch_ind])

      dict_list = [clients[u].dict for u in batch_ind]
      self.dict = dict_concatenate(dict_list, 1)

