#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import builtins
import os
import time
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import datasets.imagelistdataset
import torchvision.models as models
import warnings

from datasets.mysampler import ResumableRandomSampler
from datasets.mysampler import ResumableBatchSampler
from datasets.mysampler import ResumableDistributedSampler

import simsiam
from utils import AverageMeter, ProgressMeter, adjust_learning_rate
from utils import WindowAverageMeter, CheckpointManager

from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import save_image
import kornia as K
import SIMCLR
import AugmentationModel
import copy

elps_time = time.time()


def main_worker(gpu, ngpus_per_node, args):

    cudnn.benchmark = True
    args.environment.gpu = gpu if ngpus_per_node > 0 else None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.environment.gpu is not None:
        print("Use GPU: {} for training".format(args.environment.gpu))
        
    
    #Calculate p for transformation

    # data augmentations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(args.data.insize, scale=(0.2, 1.)),
        #transforms.RandomApply(
         #   [
          #      transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
           # ],
            #p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([simsiam.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        #normalize
    ])

    # create model
    
    colorjModel = AugmentationModel.CJAugmentationPipeline()
    for name, para in colorjModel.named_parameters():
        print(name, para.data)
    
    
    
    print("=> creating model '{}'".format(args.model.backbone.arch))
    model = simsiam.SimSiam(models.__dict__[args.model.backbone.arch],
                                    args.model.backbone.simsiam_dim,
                                    args.model.backbone.pred_dim)
    
    # print(model)
    model = model.to(device)
    if device == 'cuda':
        #net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    #if args.model.sync_bn and args.environment.gpu is not None:
     #   model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #else:
     #   warnings.warn('Original SimSiam requires sync batchnorm, ' +
      #                'you have set it to False')

    #if args.environment.gpu is not None:
        #torch.cuda.set_device(args.environment.gpu)
        #model = model.cuda(args.environment.gpu)
        #model = nn.DataParallel(model, device_ids=[0,1,2,3], output_device=args.environment.rank)
        #model = nn.DataParallel(model)
    #model.to(device)
        #raise NotImplementedError("Only DistributedDataParallel is supported.")
    # else:
    # AllGather implementation (batch shuffle, queue update, etc.) in
    # this code only supports DistributedDataParallel.
    # raise NotImplementedError("Only DistributedDataParallel is supported.")

    #print(args.model.pretrained)
    if args.model.pretrained is not None:
        ckp = torch.load(args.model.pretrained,
                         map_location='cpu')['state_dict']
        model.load_state_dict(ckp)
        print(f'Loaded {args.model.pretrained}')
        del ckp

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1)
    if args.environment.gpu is not None:
        criterion = criterion.cuda(args.environment.gpu)

    if args.optim.fix_pred_lr:
        optim_params = [{
            'params':
            model.encoder.parameters()
            if ngpus_per_node > 0 else model.encoder.parameters(),
            'fix_lr':
            False
        }, {
            'params':
            model.predictor.parameters()
            if ngpus_per_node > 0 else model.predictor.parameters(),
            'fix_lr':
            True
        }]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params,
                                args.optim.lr,
                                momentum=args.optim.momentum,
                                weight_decay=args.optim.weight_decay)

    # Create TB loggers
    writer = None
    if args.logging.log_tb and args.environment.gpu == 0:
        logdir = os.path.join(args.logging.tb_dir,
                              args.logging.name + args.logging.suffix)
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)

    cudnn.benchmark = True
    
    
    # Data loading - CIFAR10
    
    #train_set = datasets.CIFAR10(root=args.data.base_dir, train=True, download=True, transform=simsiam.TwoCropsTransform(augmentation))
    
    
    trainfname = args.data.train_filelist
    
    with open(trainfname, 'r') as f:
        filedata = f.read().splitlines()
        trainingListSize = len(filedata)
    
    print("Total Number of Training Images {}".format(trainingListSize))
    
    train_dataset = datasets.imagelistdataset.ImageListDataset(
                          trainfname,
                          base_dir=args.data.base_dir,
                          transforms=augmentation)
    
    print(f'Dataset: {len(train_dataset)}')
    
    train_sampler = ResumableRandomSampler(data_source=train_dataset)
    print(f'Sampler: {len(train_sampler)}')
    
    
    train_n_seq = args.data.n_seq_samples
    if args.model.buffer_type == 'none_noseq':
            train_n_seq = -1
    
    batch_sampler = ResumableBatchSampler(
            args.optim.batch_size,
            train_sampler,
            drop_last=True,
            n_seq_samples=train_n_seq)
    
    print(f'Batch Sampler: {len(batch_sampler)}')
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.environment.workers,
        pin_memory=True,
        prefetch_factor=1)

    # optionally resume from a checkpoint
    modules = {
        'state_dict': model,
        'optimizer': optimizer,
        'sampler': train_loader.batch_sampler
    }
    ckpt_manager = CheckpointManager(
        modules=modules,
        ckpt_dir=os.path.join(args.logging.ckpt_dir, args.logging.name),
        epoch_size=len(train_loader),
        epochs=args.optim.epochs,
        save_freq=args.logging.save_freq,
        save_freq_mints=args.logging.save_freq_mints)
    if args.environment.resume:
        args.optim.start_epoch = ckpt_manager.resume()

    # Train
    
    for epoch in range(args.optim.start_epoch, args.optim.epochs):
        print('Train Epoch {}'.format(epoch))
        sys.stdout.flush()
        batch_sampler.set_epoch(epoch=epoch)

        train(train_loader,
              model,
              colorjModel,
              criterion,
              optimizer,
              epoch,
              args,
              writer=writer,
              ckpt_manager=ckpt_manager)

        #ckpt_manager.checkpoint(epoch=epoch + 1,
         #                       save_dict={
          #                          'epoch': epoch + 1,
           #                         'batch_i': 0,
            #                        'arch': args.model.backbone.arch,
             #                   })
        batch_sampler.init_from_ckpt = False


def train(train_loader,
          model,
          colorjModel,
          criterion,
          optimizer,
          epoch,
          args,
          writer=None,
          ckpt_manager=None):

    batch_time = WindowAverageMeter('Time', fmt=':6.3f')
    data_time = WindowAverageMeter('Data', fmt=':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    lr_meter = AverageMeter('LR', ':.4e')
    buff_meters = []
    if not args.model.buffer_type.startswith('none'):
        num_seen = AverageMeter('#Seen', ':6.3f')
        num_seen_max = AverageMeter('#Seen Max', ':6.3f')
        similarity = AverageMeter('Buffer Sim', ':6.3f')
        neig_similarity = AverageMeter('Buffer Neig Sim', ':6.3f')
        buff_meters = [num_seen, num_seen_max, similarity,
        neig_similarity]
    progress = ProgressMeter(len(train_loader), [
        batch_time, data_time, lr_meter] + buff_meters + [losses]
    ,
                             prefix="Epoch: [{}]".format(epoch),
                             tbwriter=writer)
    # switch to train mode
    model.train()

    end = time.time()
    world_size = torch.distributed.get_world_size(
    ) if torch.distributed.is_initialized() else 1
    
    batchNum = 1
    for data in train_loader:
        batch_i = train_loader.batch_sampler.advance_batches_seen()

        effective_epoch = epoch + (batch_i / len(train_loader))
        lr = adjust_learning_rate(optimizer,
                                  effective_epoch,
                                  args,
                                  epoch_size=len(train_loader))
        lr_meter.update(lr)

        # measure data loading time
        images = [data['input1'], data['input2']]             
        data_time.update(time.time() - end)
        
        # send to cuda
        if args.environment.gpu is not None:
            #print(args.environment.gpu)
            images = [
                imgs.cuda(args.environment.gpu, non_blocking=True)
                for imgs in images
            ]
        images = [
                imgs.requires_grad_(True)
                for imgs in images
            ]
        # perform dynamic augmentation
        
        model.eval()
        print("Batch Number {}".format(batchNum))
        colorjModel = AugmentationModel.CJAugmentationPipeline()
        finalAugImg1, finalAugImg2 = dynamicAugmentationCheck(images[0], images[1], args, colorjModel, model)
        
        # compute output and loss
        model.train()
        p1, p2, z1, z2 = model( aug=False, x1=finalAugImg1.detach(), x2=finalAugImg2.detach())
        
        loss_per_sample = -(criterion(p1, z2.detach()) +
                            criterion(p2, z1.detach())) * 0.5
        loss = loss_per_sample.mean()
        writer.add_scalar("Loss/Batch", loss, batchNum)
        losses.update(loss.item(), images[0].size(0))
        with torch.no_grad():
            data['feature'] = torch.stack((z1, z2), 1).detach()
            data['loss'] = loss_per_sample.detach()
            if not args.model.buffer_type.startswith('none'):
                stats = train_loader.batch_sampler.update_sample_stats(data)
                if 'num_seen' in stats:
                    num_seen.update(stats['num_seen'].float().mean().item(),
                                    stats['num_seen'].shape[0])
                    num_seen_max.update(stats['num_seen'].float().max().item(),
                                        stats['num_seen'].shape[0])
                if 'similarity' in stats:
                    similarity.update(stats['similarity'].float().mean().item(),
                                      stats['similarity'].shape[0])
                if 'neighbor_similarity' in stats:
                    neig_similarity.update(
                        stats['neighbor_similarity'].float().mean().item(),
                        stats['neighbor_similarity'].shape[0])

        batchNum = batchNum + 1
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Create checkpoints
        #if ckpt_manager is not None:
            #ckpt_manager.checkpoint(epoch=epoch,
                                    #batch_i=batch_i,
                                    #save_dict={
                                        #'epoch': epoch,
                                        #'batch_i': batch_i,
                                        #'arch': args.model.backbone.arch,
                                    #})   

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log
        if batch_i % args.logging.print_freq == 0:
            tb_step = (
                epoch * len(train_loader.dataset) // args.optim.batch_size +
                batch_i * world_size)
            progress.display(batch_i)
            progress.tbwrite(tb_step)

def dynamicAugmentationCheck(x1, x2, args, colorjModel, model):
    
    for name, para in colorjModel.named_parameters():
      print(name, para.data)
    
    #First augmentation step
    aug1 = colorjModel(x1)
    aug2 = colorjModel(x2)
    
    #create a copy of original model
    
    modelCopy = copy.deepcopy(model)
    p1, p2, z1, z2 = modelCopy(aug=True, x1 = aug1, x2 = aug2)
    
    simclrLossCriterion = SIMCLR.SimCLR_Loss(batch_size = args.optim.batch_size, temperature = 0.5)
    optimizer_param = torch.optim.SGD(colorjModel.parameters(), lr=0.0001)
    
    #SIMCLR pairwise angle calculation
    batchPos, batchMinNeg = simclrLossCriterion(z1, z2)
    print("Positive angle {}".format(batchPos))
    print("Between pos and neg angles {}".format(batchMinNeg)) 
    
    cjModelLoss = nn.MSELoss() (batchPos, batchMinNeg)
    
    print("KMODEL Loss {}".format(cjModelLoss))
    
    while cjModelLoss>0.01:
        aug1 = colorjModel(x1)
        aug2 = colorjModel(x2)
        p1, p2, z1, z2 = modelCopy(aug=True, x1 = aug1, x2 = aug2) 
        
        batchPos, batchMinNeg = simclrLossCriterion(z1, z2)
        print("Positive angle {}".format(batchPos))
        print("Between pos and neg angles {}".format(batchMinNeg)) 
        cjModelLoss = nn.MSELoss() (batchPos, batchMinNeg)
        print("intermediate losses {}".format(cjModelLoss))
        optimizer_param.zero_grad()
        cjModelLoss.backward()
        optimizer_param.step()
    
    for name, para in colorjModel.named_parameters():
      print(name, para.data)
    return aug1, aug2
     
    