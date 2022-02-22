# Copyright (c) 2022-present, Js2hou.
# All rights reserved.

# Training and evaulate code.

from collections import OrderedDict
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import timm
from timm.utils import random_seed
from torch.distributed.elastic.utils.data import ElasticDistributedSampler

from dataset import build_dataset
from engine import train_one_epoch, validate
from models import YourModel
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)

    # files are saved in `args.output/tag/`
    parser.add_argument('--tag', default='default', type=str,
                        help='tag of experiment')
    parser.add_argument('--output', default='outputs',
                        help='path where to save')

    # training mode parameters
    parser.add_argument('--log-wandb', action='store_true',
                        help='Whether loggered by wandb')
    parser.add_argument('--seed', default=42)

    # distributed training parameters
    parser.add_argument('--local-rank', type=int)
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    return parser


def init_dataloader(args, trainset, valset):
    """Return train dataloader and val dataloader"""
    if args.distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        train_sampler = ElasticDistributedSampler(
            trainset, num_replicas=num_tasks, rank=global_rank)
        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, persistent_workers=True)
        val_loader = DataLoader(
            valset, batch_size=args.batch_size, num_workers=8, persistent_workers=True)
    else:
        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(
            valset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    return train_loader, val_loader


def main(args, logger):
    random_seed(args.seed + utils.get_rank())

    device_id = args.local_rank

    # wandb
    if args.log_wandb:
        import wandb
        project_path = os.path.dirname(os.path.abspath(__file__))
        _, project_name = os.path.split(project_path)
        wandb.init(project=project_name, entity='jshou', config=args)

    # load dataset
    trainset, valset, args.nb_classes = build_dataset()
    train_loader, val_loader = init_dataloader(args, trainset, valset)

    # create model
    model = timm.create_model('resnet18').cuda(
        device_id)  # model = YourModel()
    model.reset_classifier(args.nb_classes)  # 修改分类层
    model = model.cuda(device_id)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device_id])

    # create optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs)

    # used for `torchrun` in pytorch 1.9
    state = utils.load_checkpoint(model, optimizer, args.ckpt_file, device_id)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda(device_id)

    logger.info(f"Start training for {args.epochs} epochs")
    t_start = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, lr_scheduler, epoch, device_id, logger)
        val_metrics = validate(model, criterion, val_loader, device_id)

        logger.info(
            f"[Epoch {epoch}/{args.epochs}]  train loss: {train_metrics['loss']:.4f}  val loss: {val_metrics['loss']:.4f}  val acc: {val_metrics['acc1']:.2f}")

        state.epoch = epoch
        is_best = val_metrics['loss'] < state.best_state['loss']
        if is_best:
            state.best_state['loss'] = val_metrics['loss']

        if utils.is_main_process():
            utils.save_checkpoint(state, is_best, args.ckpt_file)

            if args.log_wandb:
                rowd = OrderedDict(epoch=epoch) 
                rowd.update([('train_' + k, v)
                            for k, v in train_metrics.items()])
                rowd.update([('val_' + k, v) for k, v in val_metrics.items()])
                wandb.log(rowd)

    logger.info(f"{args.tag} consumes time {time.time() - t_start:.4f} s.")
    logger.info(f"{'='*20+'End'+'='*20}\n\n")


if __name__ == '__main__':
    parser = get_args_parser()
    args, unparsed = parser.parse_known_args()

    if args.output and args.tag:
        args.output = os.path.join(args.output, args.tag)
    args.ckpt_file = os.path.join(args.output, 'ckpt_file.pth.tar')
    os.makedirs(args.output, exist_ok=True)

    # don't delete this line even when using signle gpu
    utils.init_distributed_mode(args)

    logger = utils.create_logger(args.output, args.rank)
    logger.info(f"{'='*20+'Start'+'='*20}")
    for k, v in vars(args).items():
        logger.info(f'{k}: {v}')

    main(args, logger)
