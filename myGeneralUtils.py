import argparse, torch, cv2, torch.utils.data, math
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import os

from torch.autograd import Variable
from utils import *
from XVFInet import *
from collections import Counter


def weights_init(m):
    classname = m.__class__.__name__
    if (classname.find("Conv2d")!= -1) or (classname.find("Conv3d")!= -1):
        init.xavier_normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init.zeros_(m.bias)


