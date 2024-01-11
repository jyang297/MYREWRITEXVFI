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
    parser.add_argument("--test_dir", type=str, default="./test_dir", help="test_dir path. Default at ./test_dir")
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoint_dir', help="checkpoint directory. Default at ./checkpoint_dir")
    parser.add_argument("--log_dir", type=str, default='./log_dir', help="Directory name to save training logs. Default at ./log_dir")
    parser.add_argument("--dataset", default='Vimeo', choices=['X4K1000FPS', 'Vimeo'], help="Training/test Dataset")

    parser.add_argument("--train_data_path", type=str, default="../Daasets/VIC_4K_1000FPS/train")
    parser.add_argument("--val_data_path", type=str, default="../Daasets/VIC_4K_1000FPS/val")
    parser.add_argument("--test_data_path", type=str, default="../Daasets/VIC_4K_1000FPS/test")

    parser.add_argument("--vimeo_data_path", type=str, default='./vimeo_triplet')

    """ Hyperparmeters for Training (need to set [phase='train'] (args.phase))"""
    parser.add_argument("--epochs", type=int, default=200, help='The number of epochs to run')
    parser.add_argument("freq_display", type=int, default=100, help="The number of iterations frequency for display")
    parser.add_argument("--save_img_num", type=int, default=4, help="The number of saved image while training for visualization. It should smaller than the batch_size")
    parser.add_argument("--init_lr", type=float, default=1e-4, help="The initial learning rate")
    parser.add_argument("--lr_dec_fac", type=float, default=0.25, help="step - lr_decreaseing_factor")
    parser.add_argument("--lr_milestones", type=int, default=[100, 150, 180])
    parser.add_argument("--batch_size", type=int, default=4, help="The size of batch size")
    parser.add_argument("--weight_decay", type=float, default=0, help="optim, weight decay (default=0)")
    parser.add_argument("--need_patch", default=True, help='get patch from image while training')
    parser.add_argument("--img_ch", type=int, default=3, help="the channel for image")
    parser.add_argument("--loss_type", default="L1", choices=["L1", 'MSE', "L1_Charbonnier_loss"], help="loss type")
    parser.add_argument("--S_trn", type=int, default=3, help="The lowest scale depth for training")
    parser.add_argument("S_tst", type=int, default=5, help="The lowest scale depth for test")

    """ Weighting Parameters Lambda for Losses (when [phase='train'])"""
    parser.add_argument("--rec_lambda", type=float, default=1.0, help="Lambda for Reconstruction Loss")

    """ Setting for Testing (when [phase=='test] or 'test_custom)"""
    parser.add_argument('--saving_flow_flag', default=False)
    parser.add_argument("--multiple", type=int, default=8, help="Due to the indexing problem of the file names, we recomend to use the power of 2. (2, 4, 8...)")
    parser.add_argument("--metrics_types", type=list, default=["PSNR", "SSIM", "tOF"], choices=["PSNR", "SSIM", "tOF"])

    """ Settings for test_custom (when [phase=='test_custom'])"""
    parser.add_argument("custom_path", type=str, default='./custom_path', help='path for custom video containing frames')

    return check_args(parser.parse_args)


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
    if args is None:
        exit()

    print("Exp:", args.exp_num)
    args.model_dir = args.net_type + "_" + args.dataset + "_exp" + str(args.exp_num) # ex: model_dir = XVFInet_X41000FPS_exp1

    if args.dataset == "Vimeo":
        if args.phase != "test_custom":
            args.multiple = 2
        args.S_trn = 1
        args.S_tst = 1
        args.module_scale_factor = 2
        args.patch_size = 256
        args.batch_size = 4
        print("vimeo triplet data dir : ", args.vimeo_data_path)

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

    cudnn.benchmark = True

    if args.phase == "train":
        train(model_net, criterion, device, SM, args)
        epoch = args.epochs - 1
    elif args.phase == "test" or args.phase == 'metrics_evaluation' or args.phase == 'test_custom':
        checkpoint = SM.load_odel()
        epoch = checkpoint['last_epoch']
        model_net.load_state_dict(checkpoint['state_dict_Model'])

    postfix = '_final_x' + str(args.multiple) + "_S_tst" + str(args.S_tst)

    if args.phase != "metrics_evaluation":
        print("\n------------Final Test Starts ------------\n")
        print("Evaluation on test set(final set) with multiple = %d " %(args.multiple))

        final_test_loader = get_test_data(args, multiple=args.multiple, validation=False) # multiple is only used for X4K1000FPS

        testLoss, testPSNR, testSSIM, final_pred_save_path = test(final_test_loader, model_net, criterion, epoch, args, device, multiple=args.multiple, postfix=postfix, validati=False)

        SM.write_info('Final 4k frames PSNr: {:.4}\n'.format(testPSNR))

    if args.dataset == 'X4K1000FPS' and args.phase != 'test_custom':
        final_pred_save_path = os.path.join(args.test_img_dir, args.model_dir, 'epoch_' + str(epoch).zfill(5)) + postfix
        metrics_evaluation_X_Test(final_pred_save_path, args.test_data_path, args.metrics_types, flow_flag=args.saving_flow_flag, multiple=args.multiple)


    
    print("\n------------ Test End------------\n")
    print("Exp:", args.exp_num)


    def weights_init(m):
        classname = m.__class__.__name__
        if (classname.find("Conv2d")!= -1) or (classname.find("Conv3d")!= -1):
            init.xavier_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init.zeros_(m.bias)


# get_test_data
                
# train
                
# test