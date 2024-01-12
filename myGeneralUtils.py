import argparse, torch, cv2, torch.utils.data, math
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import os

from torch.autograd import Variable
from utils import *
from myXVFInet import *
from collections import Counter

from torch.utils.tensorboard import SummaryWriter


def weights_init(m):
    classname = m.__class__.__name__
    if (classname.find("Conv2d")!= -1) or (classname.find("Conv3d")!= -1):
        init.xavier_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init.zeros_(m.bias)

"""
class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
"""


class myLogger:
    # def __init__(self, model, scheduler):
    def __init__(self, log_dir='runs/my_complex_experiment'):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, model, step):
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram(tag, value.data.cpu().numpy(), step)
            if value.grad is not None:
                self.writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), step)

    def log_images(self, tag, images, step):
        self.writer.add_images(tag, images, step)

    def log_training_loss(self, loss, step):
        self.log_scalar('Training Loss', loss, step)
    
    def log_PSNR(self, PSNR, step):
        self.log_scalar("PSNR: ", PSNR, step)

    def log_model_graph(self, model, inputs):
        self.writer.add_graph(model, inputs)
    
    def log_training_loss(self, loss, step):
        self.log_scalar('Training Loss', loss, step)

    # Add other logging methods as needed

    def close(self):
        self.writer.close()
