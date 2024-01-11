import argparse, torch, cv2, torch.utils.data, math
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import os

from torch.autograd import Variable
from utils import *
from XVFInet import *
from collections import Counter

def parse_args():
    desc = "Pytorch implementation for my XVFI"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--gpu", type=int, default=0, help="gpu index")
    parser.add_argument("--net_type", type=str, default=['XVFInet'], help="The type of my network")
    parser.add_argument("--exp_num", type=int, default=1, help="The experiment number")
    parser.add_argument("--phase", type=str, default="test", choices=['train', 'test', 'metrics_evaluation'],)
    parser.add_argument("--continue_training", action='store_true', default=False, help='continue the training')

    """ Information of directories"""
    parser.add_argument("--test_img_dir", type=str, default="./test_img_dir", help='test_img_dir path. Default at ./test_img_dir')
    parser.add_argument("test_dir", type=str, default="./test_dir", help="test_dir path. Default at ./test_dir")
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint_dir', help="checkpoint directory. Default at ./checkpoint_dir")
    parser.add_argument("--log_dir", type=str, default='./log_dir', help="Directory name to save training logs. Default at ./log_dir"

    return check_args(parser)

def check_folder(log_dir):
    if not os.path.exist(log_dir):
        os.makedirs(log_dir)
    return log_dir

def check_args(args):
    # --checkpoint dir
    check_folder(args.checkpoint_dir)

    # --text_dir
    check_folder(args.text_dir)

    # --log_dir
    check_folder(args.log_dir)

    # --test_img_dir
    check_folder(args.test_img_dir)

    return args



def main():
    args = parse_args()
    if args.dataset == "Vimeo":
        if args.phase != "test_custom":
            args.multiple = 2
        args.S_trn = 1
        args.S_tst = 1
        args.module_scale_factor = 2
        args.patch_size = 256
        args.batch_size = 4
        print("vimeo triplet data dir : ", args.vimeo_data_path)

    print("Exp:" args.exp_num)
    args.model_dir = args.net_type + "_" + args.dataset + "_exp" + str(args.exp_num)

    if args is None:
        exit()

    for arg in vars(args):
        print("# {} : {}".format(arg, getattr(args, arg)))
    device = torch.device(
        'cuda' + str(args.str(args.gpu) if torch.cuda.is_available() else 'cpu')
    )
    torch.cuda.set_device(device)

    print("Available device:" , torch.cuda.device_count())
    print("Current cuda device: ", torch.cuda.current_device())
    print("Current cuda device name: ". torch.cuda.get_device_name(device))

    if args.gpu is not None:
        print("Use GPU: {} is used.".format(args.gpu))

    SM = save_manager(args) #??

    """Initialize a model"""
    model_net = args.net_object(args).apply(weights_init).to(device)
    criterion = [set_rec_loss(args).to(device), set_smoothness_loss().to(device)]

    cudnn.benchmark``