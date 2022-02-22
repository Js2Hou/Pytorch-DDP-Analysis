# Implements your own metric here.

from collections import defaultdict
import torch
import torch.distributed as dist

from utils import is_dist_avail_and_initialized


class AverageMeter:
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

    def synchronize_between_processes(self):
        """
        Warning: only synchronize the avg!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.avg], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.avg = t[0]

    def __str__(self):
        return f'{self.avg:.4f}'


class MetricLogger():
    def __init__(self, metric_names=[], delimiter='\t'):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

        for name in metric_names:
            self.meters[name]

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, n=n)

    def add_meters(self, name, meter):
        self.meters[name] = meter

    def __str__(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(metric_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
