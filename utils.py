import os, glob, sys, torch, shutil
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from datetime import datetime
from torch.nn import init


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

class HelloWorld():
    def hello():
        print("Hellooooooo!")
        
        


