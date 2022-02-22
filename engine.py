# Copyright (c) 2022-present, Js2hou.
# All rights reserved.

# Train and eval functions used in main.py

import torch
import time
from timm.utils import accuracy

from metrics import MetricLogger


def train_one_epoch(model, criterion, data_loader, optimizer, lr_scheduler, epoch, device_id, logger):
    model.train()

    metriclogger = MetricLogger(['loss'])
    # t_start = time.time()  # test time
    for idx, batch in enumerate(data_loader):
        # logger.info(f'batch length: {len(batch[0])}')
        # t2 = time.time()
        # logger.info(f'load batch data: {t2 - t_start:.4f}')
        samples, targets = [_.cuda(device_id) for _ in batch]

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # torch.cuda.synchronize()
        metriclogger.update(n=samples.size(0), loss=loss.item())
        t3 = time.time()
        # logger.info(f'train one batch: {t3 - t2:.4f}')
        # break

    lr_scheduler.step()
    metriclogger.synchronize_between_processes()
    # t_end = time.time()  # test time
    return {k: v.avg for k, v in metriclogger.meters.items()}


@torch.no_grad()
def validate(model, criterion, data_loader, device_id):
    model.eval()

    metriclogger = MetricLogger(['loss', 'acc1', 'acc5'])

    for idx, batch in enumerate(data_loader):
        samples, targets = [_.cuda(device_id) for _ in batch]

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        # torch.cuda.synchronize()
        metriclogger.update(n=samples.size(0), loss=loss.item(),
                            acc1=acc1.item(), acc5=acc5.item())

    metriclogger.synchronize_between_processes()
    return {k: v.avg for k, v in metriclogger.meters.items()}