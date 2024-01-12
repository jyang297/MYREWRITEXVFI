import os, glob, sys, torch, shutil
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from datetime import datetime
from torch.nn import init
import datasets


def set_rec_loss(args):
    loss_type = args.loss_type
    if loss_type == "MSE":
        lossfunction = nn.MSELoss()
    elif loss_type == "L1":
        lossfunction = nn.L1Loss()
    elif loss_type == "L1_Charbonnier_loss":
        lossfunction = L1_Charbonnier_loss()
    
    return lossfunction

class L1_Charbonnier_loss(torch.nn.Module):
    def __init__(self):
        super(L1_Charbonnier_loss,self).__init__()
        self.epsilon = 1e-3
    
    def forward(self, X, Y):
        loss = torch.mean(torch.sqrt((X - Y) ** 2 + self.epsilon ** 2))
        return loss
    

class save_manager():
    def __init__(self, args):
        self.args = args
        self.model_dir = self.args.net_type + "_" + self.args.dataset + '_exp' + str(self.args.exp_num)
        print("Model_dir:", self.model_dir)

        self.checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        # check_folder(self.checkpoint_dir)

        print("Checkpoint dir :", self.checkpoint_dir)

        self.text_dir = os.path.join(self.args.text_dir, self.model_dir)
        print("Text_dir:", self.text_dir)

        """Save a text file"""
        if not os.path.exists(self.text_dir + '.txt'):
            self.log_file = open(self.text_dir + '.txt', 'w')
            self.log_file.write("------- Model Parameters -------")
            self.log_file.write(str(datetime.now()[:7] + '\n'))
            for arg in vars(self.args):
                self.log_file.write("{} : {}\n". format(arg, getattr(self.args, arg)))
                self.log_file.close()

    def write_info(self, strings):
        self.log_file = open(self.text_dir + '.txt', 'a')
        self.log_file.write(strings)
        self.log_file.close()

    def save_best_model(self, combined_state_dict, best_PSNR_flag):
        file_name = os.path.join(self.checkpoint_dir, self.model_dir + '_latest.pt')
        torch.save(combined_state_dict, file_name)
        if best_PSNR_flag:
            shutil.copyfile(file_name, os.path.join(self.checkpoint_dir, self.model_dir + '_best_PSNR.pt'))

    def save_epc_model(self, combined_state_dict, epoch):
        file_name = os.path.join(self.checkpoint_dir, self.model_dir + '_epc' + str(epoch) + '.pt')
        torch.save(combined_state_dict, file_name)

    def load_epc_model(self, epoch):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.model_dir + '_epc' + str(epoch - 1) + '.pt'))
        print("Load model '{}', epoch: {}, best PSNR: {:3f}".format())

    def load_model(self, ):
        # checkpoint = torch.load(self.checkpoint_dir + '/' + self.model_dir + '_latest.pt')
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.model_dir + '_latest.pt'), map_location='cuda:0')
        print("load model '{}', epoch: {},".format(
            os.path.join(self.checkpoint_dir, self.model_dir + '_latest.pt'), checkpoint['last_epoch'] + 1))
        return checkpoint

    def load_best_PSNR_model(self, ):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.model_dir + '_best_PSNR.pt'))
        print("load _best_PSNR model '{}', epoch: {}, best_PSNR: {:3f}, best_SSIM: {:3f}".format(
            os.path.join(self.checkpoint_dir, self.model_dir + '_best_PSNR.pt'), checkpoint['last_epoch'] + 1,
            checkpoint['best_PSNR'], checkpoint['best_SSIM']))
        return checkpoint

# def get_test_data(args, multiple, validation):
#    if args.dataset == "X4K1000FPS" and args.phase != 'test_custom':
        # data_test = X_Test()



        
    
    


class HelloWorld():
    def hello():
        print("Hellooooooo!")



        
        
        


